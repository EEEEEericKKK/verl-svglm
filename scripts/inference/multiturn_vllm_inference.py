#!/usr/bin/env python
"""
Multi-modal, Multi-turn Tool Use Inference with vLLM

This script performs inference on datasets with multi-turn conversations,
supporting tool use (SVG diagram generation) and multi-modal inputs.
"""

import argparse
import io
import json
import logging
import multiprocessing
import os
import re
import tempfile
from multiprocessing import Process
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import cairosvg
import pandas as pd
from PIL import Image
from qwen_vl_utils import fetch_image, process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams

# Prevent multiprocessing issues with CUDA
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# System Prompt Configuration
# ============================================================================

prompt_path = "/proj/inf-scaling/csl/svglm/data/system_prompt_svgtool.txt"
with open(prompt_path, "r") as f:
    SYSTEM_PROMPT_WITH_TOOL = f.read().strip()

SYSTEM_PROMPT_WITHOUT_TOOL = (
    "You are an expert in solving plane geometry reasoning problems. "
    "Your final answer should be presented as: \\boxed{answer}"
)


def get_system_prompt(enable_tool_call: bool = True) -> str:
    """Get the appropriate system prompt based on tool call setting."""
    return SYSTEM_PROMPT_WITH_TOOL if enable_tool_call else SYSTEM_PROMPT_WITHOUT_TOOL


# ============================================================================
# Image Processing Utilities
# ============================================================================

def process_image(image: dict | Image.Image, image_patch_size: int = 14) -> Image.Image:
    """Process image for vision model, converting from various formats to PIL Image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    
    if "bytes" in image:
        assert "image" not in image, "Cannot have both 'bytes' and 'image' keys"
        image["image"] = Image.open(io.BytesIO(image["bytes"]))
    
    return fetch_image(image, image_patch_size=image_patch_size)


def build_messages(
    messages: List[Dict[str, Any]],
    images: List[Any],
    videos: List[Any] = None,
    image_patch_size: int = 14
) -> List[Dict[str, Any]]:
    """
    Replace <image> and <video> placeholders in messages with actual media.
    
    This follows the approach from multiturn_sft_dataset.py.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        images: List of images (bytes dicts or PIL Images)
        videos: List of videos (optional)
        image_patch_size: Patch size for image processing
    
    Returns:
        List of processed messages with media embedded
    """
    videos = videos or []
    image_offset, video_offset = 0, 0
    processed_messages = []
    
    for message in messages:
        role = message["role"]
        content = message["content"]
        
        # Already in list format or no placeholders
        if not isinstance(content, str) or ("<image>" not in content and "<video>" not in content):
            processed_messages.append(message)
            continue
        
        # Process placeholders
        content_list = []
        segments = re.split(r"(<image>|<video>)", content)
        
        for segment in segments:
            if not segment:
                continue
            elif segment == "<image>" and image_offset < len(images):
                img = process_image(images[image_offset], image_patch_size=image_patch_size)
                content_list.append({"type": "image", "image": img})
                image_offset += 1
            elif segment == "<video>" and video_offset < len(videos):
                content_list.append({"type": "video", "video": videos[video_offset]})
                video_offset += 1
            elif segment.strip():
                content_list.append({"type": "text", "text": segment})
        
        if content_list:
            processed_messages.append({
                "role": role,
                "content": content_list if len(content_list) > 1 else content_list[0]
            })
        else:
            processed_messages.append(message)
    
    if image_offset != len(images):
        logger.warning(f"Not all images used: {image_offset}/{len(images)}")
    if video_offset != len(videos):
        logger.warning(f"Not all videos used: {video_offset}/{len(videos)}")
    
    return processed_messages


# ============================================================================
# SVG Tool Implementation
# ============================================================================

class SVGTool:
    """Tool for converting SVG code to PNG images for geometric diagrams."""
    
    def __init__(self):
        self._temp_dir = tempfile.mkdtemp()
        self._instances = {}
    
    def get_schema(self) -> Dict[str, Any]:
        """Return OpenAI-compatible tool schema."""
        return {
            "type": "function",
            "function": {
                "name": "svg_tool",
                "description": "Generates geometric diagrams from SVG code",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "svg_str": {
                            "type": "string",
                            "description": "SVG code for the diagram (use 1000x1000 viewBox)",
                        },
                        "task_type": {
                            "type": "string",
                            "description": "Either 'full_reconstruction' or 'auxiliary'",
                            "enum": ["full_reconstruction", "auxiliary"],
                        },
                    },
                    "required": ["svg_str", "task_type"],
                },
            }
        }
    
    def create_instance(self, instance_id: Optional[str] = None) -> str:
        """Create a new tool instance for a conversation."""
        if instance_id is None:
            instance_id = str(uuid4())
        self._instances[instance_id] = {"images": []}
        return instance_id
    
    def store_initial_images(self, instance_id: str, images: List[Image.Image]):
        """Store initial images for auxiliary task overlays."""
        if instance_id in self._instances:
            self._instances[instance_id]["images"].extend(images)
    
    def execute(
        self,
        instance_id: str,
        svg_str: str,
        task_type: str
    ) -> tuple[Optional[Image.Image], str, bool]:
        """
        Render SVG to PNG image.
        
        Returns:
            (image, message, success)
        """
        if task_type not in ["full_reconstruction", "auxiliary"]:
            return None, f"Invalid task_type: {task_type}", False
        
        try:
            # Render SVG to PNG
            svg_png_bytes = cairosvg.svg2png(bytestring=svg_str.encode('utf-8'))
            svg_img = Image.open(io.BytesIO(svg_png_bytes)).convert('RGBA')
            svg_width, svg_height = svg_img.size
            
            # Select background based on task type
            if task_type == "auxiliary":
                images = self._instances.get(instance_id, {}).get("images", [])
                if images:
                    # Use first image as background
                    bg = images[0].convert('RGB')
                    bg = self._resize_and_fit(bg, svg_width, svg_height)
                else:
                    bg = Image.new('RGB', (svg_width, svg_height), 'white')
            else:
                bg = Image.new('RGB', (svg_width, svg_height), 'white')
            
            # Composite SVG over background
            bg.paste(svg_img, (0, 0), svg_img)
            processed = process_image(bg)
            
            # Store generated image
            self._instances[instance_id]["images"].append(processed)
            
            return processed, "Here is the rendered image.", True
            
        except Exception as e:
            error_msg = f"Error rendering SVG: {str(e)}"
            logger.error(error_msg)
            return None, error_msg, False
    
    def _resize_and_fit(
        self,
        img: Image.Image,
        target_width: int,
        target_height: int
    ) -> Image.Image:
        """Resize image to fit target dimensions, centering if needed."""
        orig_width, orig_height = img.size
        
        # Calculate new dimensions maintaining aspect ratio
        if orig_width > orig_height:
            new_width = target_width
            new_height = int(orig_height * target_width / orig_width)
        else:
            new_height = target_height
            new_width = int(orig_width * target_height / orig_height)
        
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Center on white canvas
        if new_width < target_width or new_height < target_height:
            canvas = Image.new('RGB', (target_width, target_height), 'white')
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            canvas.paste(img, (x_offset, y_offset))
            return canvas
        elif new_width > target_width or new_height > target_height:
            x_offset = (new_width - target_width) // 2
            y_offset = (new_height - target_height) // 2
            return img.crop((x_offset, y_offset, x_offset + target_width, y_offset + target_height))
        
        return img
    
    def release(self, instance_id: str):
        """Release tool instance resources."""
        self._instances.pop(instance_id, None)


# ============================================================================
# Conversation Utilities
# ============================================================================

def extract_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Extract tool call JSON from assistant message."""
    tool_tag_pairs = [
        ['<tool_call>', '</tool_call>'],
        ['<tool>', '</tool>']
    ]
    
    for start_tag, end_tag in tool_tag_pairs:
        start = text.find(start_tag)
        if start != -1:
            end = text.find(end_tag, start + len(start_tag))
            if end != -1:
                try:
                    payload_str = text[start + len(start_tag):end]
                    return json.loads(payload_str)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse tool call: {e}")
                    return None
    return None


def extract_thinking(text: str) -> tuple[str, str]:
    """Extract thinking content from <think> tags."""
    start_tag, end_tag = '<think>', '</think>'
    start = text.find(start_tag)
    end = text.find(end_tag, start + len(start_tag)) if start != -1 else -1
    
    if start == -1 or end == -1:
        return "", text
    
    thinking = text[start + len(start_tag):end]
    text_without = text[:start] + text[end + len(end_tag):]
    return thinking.strip(), text_without.strip()


def extract_images_from_messages(messages: List[Dict[str, Any]]) -> List[Image.Image]:
    """Extract PIL Images from message content."""
    images = []
    for msg in messages:
        content = msg.get('content', [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'image':
                    img = item.get('image')
                    if isinstance(img, Image.Image):
                        images.append(img)
    return images


# ============================================================================
# Inference Engine
# ============================================================================

class MultiTurnInferenceEngine:
    """Multi-turn inference with vLLM and tool use support."""
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 8192,
        max_num_seqs: int = 32,
        enable_prefix_caching: bool = True,
        enable_tool_call: bool = True,
    ):
        self.model_path = model_path
        self.enable_tool_call = enable_tool_call
        self.tool = SVGTool()
        
        logger.info(f"Loading tokenizer and processor from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        # Detect model type for proper multimodal handling
        config_path = Path(model_path) / "config.json"
        self.is_qwen_model = False
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                model_type = config.get("model_type", "").lower()
                self.is_qwen_model = "qwen" in model_type
        logger.info(f"Model type detection: is_qwen_model={self.is_qwen_model}")
        
        logger.info("Initializing vLLM engine")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            enable_prefix_caching=enable_prefix_caching,
            limit_mm_per_prompt={"image": 10},
        )
        logger.info("Initialization complete")
    
    def generate_response(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
    ) -> str:
        """Generate a single response from the model."""
        # Convert messages for chat template: tool messages become user messages
        # to maintain user/assistant alternation
        filtered_messages = []
        last_role = None
        system_prompt = None
        
        for msg in messages:
            role = msg.get('role')
            
            # Extract system message but don't add it yet (LLaVA doesn't support system role)
            if role == 'system':
                if system_prompt is None:
                    system_prompt = msg.get('content', '')
                continue
            
            # Convert tool to user
            if role == 'tool':
                role = 'user'
                msg = {'role': 'user', 'content': msg.get('content', '')}
            
            # Skip consecutive messages with the same role to maintain alternation
            if role in ['user', 'assistant']:
                if role != last_role or role == 'user':  # Allow consecutive user messages
                    filtered_messages.append(msg)
                    last_role = role
        
        # Ensure we end with a user message if we're adding generation prompt
        if filtered_messages and filtered_messages[-1].get('role') == 'assistant':
            # This shouldn't happen if we're calling generate_response correctly
            logger.warning("Last message is assistant, which is unusual for generation")
        
        # Handle chat template differently based on model type
        if self.is_qwen_model:
            # Qwen models: Keep multimodal structure as-is for chat template
            # Qwen's chat template natively handles list-format content with images
            template_messages = filtered_messages
        else:
            # LLaVA models: Convert list-format content to string with <image> placeholders
            # LLaVA expects string content with placeholders for multimodal
            template_messages = []
            for msg in filtered_messages:
                msg_copy = msg.copy()
                content = msg_copy.get('content')
                if isinstance(content, list):
                    # Convert multimodal list to string with <image> placeholders
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get('type') == 'text':
                                text_parts.append(item.get('text', ''))
                            elif item.get('type') == 'image':
                                text_parts.append('<image>')
                            elif item.get('type') == 'video':
                                text_parts.append('<video>')
                    msg_copy['content'] = ''.join(text_parts) if text_parts else ''
                template_messages.append(msg_copy)
        
        prompt = self.tokenizer.apply_chat_template(
            template_messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        image_inputs, _ = process_vision_info(messages)
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id],
            extra_args={"enable_thinking": True}
        )
        
        if image_inputs:
            outputs = self.llm.generate(
                {"prompt": prompt, "multi_modal_data": {"image": image_inputs}},
                sampling_params=sampling_params
            )
        else:
            outputs = self.llm.generate(prompt, sampling_params=sampling_params)
        
        return outputs[0].outputs[0].text
    
    def run_conversation(
        self,
        initial_messages: List[Dict[str, Any]],
        max_turns: int = 10,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        output_dir: Optional[str] = None,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Run multi-turn conversation with tool use.
        
        Returns:
            (conversation, metadata)
        """
        messages = initial_messages.copy()
        instance_id = self.tool.create_instance()
        
        # Store initial images for auxiliary task
        initial_images = extract_images_from_messages(initial_messages)
        if initial_images:
            self.tool.store_initial_images(instance_id, initial_images)
        
        metadata = {
            "turns": 0,
            "tool_calls": 0,
            "successful_tool_calls": 0,
            "thinking_detected": False,
        }
        
        for turn in range(max_turns):
            response = self.generate_response(
                messages, temperature=temperature, top_p=top_p, max_tokens=max_tokens
            )
            metadata["turns"] += 1
            
            # Check for thinking tags
            thinking, _ = extract_thinking(response)
            if thinking:
                metadata["thinking_detected"] = True
            
            # Check for tool call (only if tool calls are enabled)
            tool_call = extract_tool_call(response) if self.enable_tool_call else None
            
            if tool_call:
                metadata["tool_calls"] += 1
                messages.append({"role": "assistant", "content": response})
                
                # Execute tool
                function_name = tool_call.get("name")
                arguments = tool_call.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                
                if function_name == "svg_tool":
                    svg_str = arguments.get("svg_str", "")
                    task_type = arguments.get("task_type", "full_reconstruction")
                    
                    image, message, success = self.tool.execute(instance_id, svg_str, task_type)
                    
                    if success:
                        metadata["successful_tool_calls"] += 1
                        tool_msg = {"role": "tool", "content": [{"type": "text", "text": message}]}
                        
                        if image and output_dir:
                            os.makedirs(output_dir, exist_ok=True)
                            img_filename = f"generated_{metadata['successful_tool_calls']}_t{turn}_{task_type}.png"
                            img_path = os.path.join(output_dir, img_filename)
                            image.save(img_path)
                            tool_msg["content"].append({"type": "image", "image": image})
                        
                        messages.append(tool_msg)
                    else:
                        messages.append({"role": "tool", "content": message})
                else:
                    messages.append({"role": "user", "content": f"Unknown tool: {function_name}"})
            else:
                # Final response
                messages.append({"role": "assistant", "content": response})
                break
        
        self.tool.release(instance_id)
        return messages, metadata


# ============================================================================
# Dataset Processing
# ============================================================================

def convert_nested_to_list(data):
    """Recursively convert numpy arrays to lists."""
    if isinstance(data, dict):
        return {k: convert_nested_to_list(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_nested_to_list(elem) for elem in data]
    elif hasattr(data, 'tolist'):
        return convert_nested_to_list(data.tolist())
    return data


def load_dataset(data_path: str) -> List[Dict[str, Any]]:
    """Load dataset from parquet file."""
    if not data_path.endswith('.parquet'):
        raise ValueError(f"Only parquet format supported, got: {data_path}")
    
    df = pd.read_parquet(data_path)
    return [convert_nested_to_list(row.to_dict()) for _, row in df.iterrows()]


def serialize_sample(sample: Dict[str, Any], exclude_keys: List[str]) -> Dict[str, Any]:
    """Create JSON-serializable version of sample, excluding certain keys."""
    def is_serializable(obj):
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return True
        if isinstance(obj, (list, tuple)):
            return all(is_serializable(item) for item in obj)
        if isinstance(obj, dict):
            return all(isinstance(k, str) and is_serializable(v) for k, v in obj.items())
        return False
    
    result = {}
    for k, v in sample.items():
        if k not in exclude_keys and is_serializable(v):
            result[k] = v
    return result


def serialize_conversation(conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert conversation to JSON-serializable format, excluding images."""
    result = []
    for msg in conversation:
        serialized = {"role": msg["role"]}
        content = msg["content"]
        
        if isinstance(content, str):
            serialized["content"] = content
        elif isinstance(content, list):
            # Keep only text items
            text_items = [item for item in content if item.get("type") == "text"]
            serialized["content"] = text_items
        else:
            serialized["content"] = content
        
        result.append(serialized)
    return result


# ============================================================================
# Dataset Format Handlers
# ============================================================================

def prepare_messages_sft(
    sample: Dict[str, Any],
    messages_key: str,
    sample_id: str,
    ground_truth_key: str = "answer",
    keep_assistant: str = "last",
    **kwargs
) -> tuple[List[Dict[str, Any]], Optional[Any]]:
    """
    Prepare SFT format messages.
    
    For SFT datasets:
    - Extract ground truth from 'answer' or 'ground_truth' field
    - Exclude assistant message (first or last) and everything after
    """
    messages = sample.get(messages_key, [])
    
    if not messages:
        logger.warning(f"Sample {sample_id}: No messages found")
        return [], None
    
    # Extract ground truth from dataset
    ground_truth = sample.get(ground_truth_key)
    if ground_truth is None:
        # Try alternative key
        alt_key = "ground_truth" if ground_truth_key == "answer" else "answer"
        ground_truth = sample.get(alt_key)
    
    # Exclude assistant message based on keep_assistant setting
    if keep_assistant == "first":
        # Find first assistant message and exclude it and everything after
        first_assistant_idx = -1
        for idx, msg in enumerate(messages):
            if msg.get('role') == 'assistant':
                first_assistant_idx = idx
                break
        
        if first_assistant_idx != -1:
            logger.info(f"Sample {sample_id}: Excluding first assistant message at index {first_assistant_idx}")
            messages = messages[:first_assistant_idx]
    else:  # keep_assistant == "last"
        # Find last assistant message and exclude it and everything after
        last_assistant_idx = -1
        for idx, msg in enumerate(messages):
            if msg.get('role') == 'assistant':
                last_assistant_idx = idx
        
        if last_assistant_idx != -1:
            logger.info(f"Sample {sample_id}: Excluding last assistant message at index {last_assistant_idx}")
            messages = messages[:last_assistant_idx]
    
    return messages, ground_truth


def prepare_messages_mathcanvas(
    sample: Dict[str, Any],
    messages_key: str,
    sample_id: str,
    ground_truth_key: str = "answer",
    keep_assistant: str = "last",
    **kwargs
) -> tuple[List[Dict[str, Any]], Optional[Any]]:
    """
    Prepare MathCanvas format messages.
    
    For MathCanvas datasets:
    - Extract question from 'question_interleave' field
    - Build a single user message from interleaved text/image content
    - Extract ground truth from 'answer' field
    """
    # Get question interleave content
    question_interleave = sample.get('question_interleave', [])
    if keep_assistant == "last":
        question_interleave += sample.get('solution_interleave', [])[:-1]
    
    # Parse if it's a string representation of list
    if isinstance(question_interleave, str):
        import ast
        try:
            question_interleave = ast.literal_eval(question_interleave)
        except:
            logger.warning(f"Sample {sample_id}: Failed to parse question_interleave")
            return [], None
    
    if not question_interleave:
        logger.warning(f"Sample {sample_id}: No question_interleave found")
        return [], None
    
    # Build user message content from interleaved parts
    # MathCanvas format has text and image parts interleaved
    content_str = ""
    for part in question_interleave:
        if isinstance(part, dict):
            part_type = part.get('type', 'text')
            part_content = part.get('content', '')
            
            if part_type == 'text':
                # content_parts.append({"type": "text", "text": part_content})
                content_str += part_content
            elif part_type == 'image':
                # Image reference - actual images are in 'question_images' field
                # We'll add a placeholder, images will be handled separately
                content_str += "<image>"
    
    # Create user message
    messages = [
        {
            "role": "user",
            "content": content_str
        }
    ]
    
    # Extract ground truth
    ground_truth = sample.get(ground_truth_key, sample.get('answer'))
    
    return messages, ground_truth


# Add more format preparation functions here as needed
# def prepare_messages_custom_format(sample, messages_key, sample_id, **kwargs):
#     """Prepare messages for custom format."""
#     pass


def prepare_messages(
    dataset_format: str,
    sample: Dict[str, Any],
    messages_key: str,
    sample_id: str,
    **kwargs
) -> tuple[List[Dict[str, Any]], Optional[Any]]:
    """
    Prepare messages for inference based on dataset format.
    
    Args:
        dataset_format: Format name (e.g., "sft", "custom")
        sample: Raw sample from dataset
        messages_key: Key containing messages
        sample_id: Sample identifier for logging
        **kwargs: Format-specific parameters
    
    Returns:
        (prepared_messages, ground_truth)
    """
    format_name = dataset_format.lower()
    
    if format_name == "sft":
        return prepare_messages_sft(sample, messages_key, sample_id, **kwargs)
    elif format_name == "mathcanvas":
        return prepare_messages_mathcanvas(sample, messages_key, sample_id, **kwargs)
    # Add more formats here
    # elif format_name == "custom_format":
    #     return prepare_messages_custom_format(sample, messages_key, sample_id, **kwargs)
    else:
        raise ValueError(
            f"Unknown dataset format: {dataset_format}. "
            f"Available formats: ['sft', 'mathcanvas']"
        )


# ============================================================================
# Sample Processing
# ============================================================================

def process_sample(
    sample: Dict[str, Any],
    engine: MultiTurnInferenceEngine,
    messages_key: str,
    image_key: str,
    video_key: str,
    image_patch_size: int,
    max_turns: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    output_dir: Optional[str],
    sample_id: str,
    enable_tool_call: bool = True,
    dataset_format: str = "sft",
    ground_truth_key: str = "answer",
    keep_assistant: str = "last",
) -> Optional[Dict[str, Any]]:
    """Process a single sample from dataset."""
    if image_key == "question_images" and keep_assistant == "last":
        images = sample.get("question_images", []) + sample.get("solution_images", [])
    else:
        images = sample.get(image_key, [])
    videos = sample.get(video_key, [])
    
    # Prepare messages based on dataset format
    messages, ground_truth = prepare_messages(
        dataset_format=dataset_format,
        sample=sample,
        messages_key=messages_key,
        sample_id=sample_id,
        ground_truth_key=ground_truth_key,
        keep_assistant=keep_assistant,
    )
    
    if not messages:
        return None
    
    # Build messages with media
    processed_messages = build_messages(messages, images, videos, image_patch_size)
    
    # Remove any existing system messages
    processed_messages = [msg for msg in processed_messages if msg.get('role') != 'system']
    
    # Prepend system prompt to first user message (LLaVA-compatible)
    system_prompt = get_system_prompt(enable_tool_call)
    if processed_messages:
        for idx, msg in enumerate(processed_messages):
            if msg.get('role') == 'user':
                content = msg.get('content')
                # Handle both string and list content
                if isinstance(content, str):
                    processed_messages[idx]['content'] = f"{system_prompt}\n\n{content}"
                elif isinstance(content, list):
                    # Prepend system prompt to first text element
                    modified_content = []
                    system_added = False
                    for item in content:
                        if not system_added and isinstance(item, dict) and item.get('type') == 'text':
                            modified_item = item.copy()
                            modified_item['text'] = f"{system_prompt}\n\n{item.get('text', '')}"
                            modified_content.append(modified_item)
                            system_added = True
                        else:
                            modified_content.append(item)
                    # If no text element found, prepend new text element
                    if not system_added:
                        modified_content = [{'type': 'text', 'text': system_prompt}] + content
                    processed_messages[idx]['content'] = modified_content
                break
    
    # Save input images
    sample_output_dir = None
    if output_dir:
        sample_output_dir = os.path.join(output_dir, sample_id)
        os.makedirs(sample_output_dir, exist_ok=True)
        
        for img_idx, img in enumerate(extract_images_from_messages(processed_messages)):
            img.save(os.path.join(sample_output_dir, f"input_{img_idx}.png"))
    
    # Run inference
    conversation, metadata = engine.run_conversation(
        processed_messages,
        max_turns=max_turns,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        output_dir=sample_output_dir
    )
    
    # Prepare result
    serialized_sample = serialize_sample(sample, [image_key, video_key])
    serialized_sample['num_images'] = len(images)
    serialized_sample['num_videos'] = len(videos)
    
    result = {
        "original_sample": serialized_sample,
        "generated_conversation": serialize_conversation(conversation),
        "metadata": metadata
    }
    
    # Add ground truth if available
    if ground_truth is not None:
        result["ground_truth"] = ground_truth
    
    return result


# ============================================================================
# Multi-GPU Processing
# ============================================================================

def worker_process(gpu_id: int, args, dataset_chunk: List[Dict[str, Any]], output_path: str):
    """Worker process for processing a chunk of dataset on a specific GPU."""
    import sys
    
    # Set CUDA device for this worker
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Update args for this worker
    args.gpu_id = gpu_id
    args.output_path = output_path
    
    # Initialize engine
    engine = MultiTurnInferenceEngine(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_tool_call=args.enable_tool_call,
    )
    
    # Setup output
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    output_basename = os.path.splitext(os.path.basename(output_path))[0]
    images_dir = os.path.join(output_dir, f"{output_basename}_images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Process samples
    logger.info(f"GPU {gpu_id}: Processing {len(dataset_chunk)} samples")
    results = []
    
    with open(output_path, 'w') as f:
        for idx, sample in enumerate(tqdm(dataset_chunk, desc=f"GPU {gpu_id}", position=gpu_id)):
            sample_id = str(sample.get('id', sample.get('problem_id', f"sample_{idx:05d}")))
            
            result = process_sample(
                sample=sample,
                engine=engine,
                messages_key=args.messages_key,
                image_key=args.image_key,
                video_key=args.video_key,
                image_patch_size=args.image_patch_size,
                max_turns=args.max_turns,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                output_dir=images_dir,
                sample_id=sample_id,
                enable_tool_call=args.enable_tool_call,
                dataset_format=args.dataset_format,
                ground_truth_key=args.ground_truth_key,
                keep_assistant=args.keep_assistant,
            )
            
            if result:
                f.write(json.dumps(result) + '\n')
                f.flush()
                results.append(result)
    
    logger.info(f"GPU {gpu_id}: Completed {len(results)} samples")


def run_multi_gpu_inference(args, dataset: List[Dict[str, Any]]):
    """Run inference across multiple GPUs using multiprocessing."""
    # Set spawn method for CUDA compatibility
    multiprocessing.set_start_method('spawn', force=True)
    
    num_gpus = args.num_gpus
    chunk_size = len(dataset) // num_gpus
    
    # Split dataset into chunks
    chunks = []
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size if i < num_gpus - 1 else len(dataset)
        chunks.append(dataset[start_idx:end_idx])
    
    logger.info(f"Split {len(dataset)} samples into {num_gpus} chunks")
    for i, chunk in enumerate(chunks):
        logger.info(f"  GPU {i}: {len(chunk)} samples")
    
    # Create output paths for each worker
    output_dir = os.path.dirname(args.output_path)
    output_basename = os.path.splitext(os.path.basename(args.output_path))[0]
    output_ext = os.path.splitext(args.output_path)[1]
    
    worker_outputs = []
    for i in range(num_gpus):
        worker_output = os.path.join(output_dir, f"{output_basename}_gpu{i}{output_ext}")
        worker_outputs.append(worker_output)
    
    # Start worker processes
    processes = []
    for gpu_id, (chunk, output_path) in enumerate(zip(chunks, worker_outputs)):
        p = Process(
            target=worker_process,
            args=(gpu_id, args, chunk, output_path)
        )
        p.start()
        processes.append(p)
        logger.info(f"Started worker process for GPU {gpu_id}")
    
    # Wait for all processes to complete
    for gpu_id, p in enumerate(processes):
        p.join()
        logger.info(f"Worker process for GPU {gpu_id} completed")
    
    # Merge output files
    logger.info(f"Merging output files into {args.output_path}")
    with open(args.output_path, 'w') as outfile:
        for worker_output in worker_outputs:
            if os.path.exists(worker_output):
                with open(worker_output, 'r') as infile:
                    for line in infile:
                        outfile.write(line)
                # Remove temporary worker output
                os.remove(worker_output)
    
    logger.info(f"Multi-GPU inference complete. Results saved to {args.output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-modal multi-turn inference with vLLM and tool use"
    )
    
    # Model config
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=True)
    
    # Multi-GPU config
    parser.add_argument("--num_gpus", type=int, default=1,
                       help="Number of GPUs to use for data parallelism (one process per GPU)")
    parser.add_argument("--gpu_id", type=int, default=None,
                       help="Specific GPU ID to use (for internal use by multiprocessing)")
    
    # Data config
    parser.add_argument("--data_path", type=str, required=True, help="Path to parquet file")
    parser.add_argument("--output_path", type=str, required=True, help="Output jsonl file")
    parser.add_argument("--messages_key", type=str, default="messages")
    parser.add_argument("--image_key", type=str, default="images")
    parser.add_argument("--video_key", type=str, default="videos")
    parser.add_argument("--image_patch_size", type=int, default=14)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--dataset_format", type=str, default="sft",
                       help="Dataset format. Available: ['sft', 'mathcanvas']")
    parser.add_argument("--ground_truth_key", type=str, default="answer",
                       help="Key for ground truth answer (will try 'answer' and 'ground_truth')")
    parser.add_argument("--keep_assistant", type=str, default="last", choices=["first", "last"],
                       help="Which assistant message to generate from: 'first' excludes first assistant, 'last' excludes last assistant")
    
    # Generation config
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--enable_tool_call", action="store_true", default=False,
                       help="Enable tool call support (SVG generation)")
    
    args = parser.parse_args()
    
    # Validate dataset format
    try:
        # Test if format is valid
        prepare_messages(args.dataset_format, {}, "", "", ground_truth_key=args.ground_truth_key)
    except ValueError as e:
        parser.error(str(e))
    except Exception:
        pass  # Expected to fail on empty sample, just checking format name
    
    # Load dataset
    logger.info(f"Loading dataset from {args.data_path}")
    dataset = load_dataset(args.data_path)
    if args.num_samples:
        dataset = dataset[:args.num_samples]
    logger.info(f"Loaded {len(dataset)} samples")
    logger.info(f"Using dataset format: {args.dataset_format}")
    
    # Multi-GPU processing
    if args.num_gpus > 1 and args.gpu_id is None:
        logger.info(f"Using {args.num_gpus} GPUs for parallel inference")
        run_multi_gpu_inference(args, dataset)
        return
    
    # Single GPU processing (or specific GPU worker)
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        logger.info(f"Worker using GPU {args.gpu_id}")
    
    # Initialize engine
    engine = MultiTurnInferenceEngine(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_tool_call=args.enable_tool_call,
    )
    
    # Setup output
    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    output_basename = os.path.splitext(os.path.basename(args.output_path))[0]
    images_dir = os.path.join(output_dir, f"{output_basename}_images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Process samples
    logger.info("Starting inference...")
    results = []
    
    with open(args.output_path, 'w') as f:
        for idx, sample in enumerate(tqdm(dataset, desc="Processing")):
            sample_id = str(sample.get('id', sample.get('problem_id', f"sample_{idx:05d}")))
            
            result = process_sample(
                sample=sample,
                engine=engine,
                messages_key=args.messages_key,
                image_key=args.image_key,
                video_key=args.video_key,
                image_patch_size=args.image_patch_size,
                max_turns=args.max_turns,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                output_dir=images_dir,
                sample_id=sample_id,
                enable_tool_call=args.enable_tool_call,
                dataset_format=args.dataset_format,
                ground_truth_key=args.ground_truth_key,
                keep_assistant=args.keep_assistant,
            )
            
            if result:
                f.write(json.dumps(result) + '\n')
                f.flush()
                results.append(result)
    
    # Print summary
    if results:
        total_turns = sum(r["metadata"]["turns"] for r in results)
        total_calls = sum(r["metadata"]["tool_calls"] for r in results)
        success_calls = sum(r["metadata"]["successful_tool_calls"] for r in results)
        thinking_count = sum(1 for r in results if r["metadata"]["thinking_detected"])
        
        logger.info("\n" + "="*80)
        logger.info("Inference Summary")
        logger.info("="*80)
        logger.info(f"Samples processed: {len(results)}")
        logger.info(f"Total turns: {total_turns} (avg: {total_turns/len(results):.2f})")
        logger.info(f"Tool calls: {total_calls} (successful: {success_calls})")
        if total_calls > 0:
            logger.info(f"Success rate: {100*success_calls/total_calls:.1f}%")
        logger.info(f"Samples with thinking: {thinking_count} ({100*thinking_count/len(results):.1f}%)")
        logger.info(f"Results saved to: {args.output_path}")
        logger.info("="*80)


if __name__ == "__main__":
    main()
