#!/usr/bin/env python
# Copyright 2025
# Multi-modal, Multi-turn Tool Use Inference with vLLM for Geometry3k Dataset

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import cairosvg
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# System Message
# ============================================================================


def get_system_message():
    # return "You are an expert in solving logical reasoning problems with visualizations. Your final answer should be formatted as: <answer> Final answer: xxx </answer>. You should replace xxx with your final answer."
    # """Extract the system message from the SVGAgent."""
    return (
        "You are a geometry and visualization assistant that solves problems by generating SVG diagrams "
        "and inspecting their rendered images.\n\n"
        "You MUST strictly follow this response structure:\n"
        "1) First, output a <think>...</think> block.\n"
        "2) After the first <think>, you MAY call the svg_to_png tool (SVG code ONLY in the tool arguments).\n"
        "3) After receiving the rendered image, you MUST output another <think>...</think> block that inspects the image.\n"
        "4) If anything is unclear or incorrect, refine the SVG and call svg_to_png again. You may call svg_to_png multiple times.\n"
        "5) Finally, output an <answer>...</answer> block with step-by-step solution and final result.\n\n"
        "Additional rules:\n"
        "- Every svg_to_png tool call MUST be preceded by a <think>...</think> block.\n"
        "- Between any tool call and the final <answer>, there MUST be at least one <think>...</think> block.\n"
        "- Do NOT put SVG code inside <think> or <answer>; SVG code only appears inside the tool call arguments.\n"
        "- The final answer must ONLY appear inside <answer>.\n"
    )


# ============================================================================
# SVG to PNG Tool (copied from verl/tools/svg_to_png_tool.py)
# ============================================================================


def process_image(image: dict | Image.Image, image_patch_size: int = 14) -> Image.Image:
    """Process image for vision model."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    
    if "bytes" in image:
        assert "image" not in image, "Cannot have both `bytes` and `image`"
        from io import BytesIO
        image["image"] = Image.open(BytesIO(image["bytes"]))
    
    from qwen_vl_utils import fetch_image
    return fetch_image(image, image_patch_size=image_patch_size)


class SvgToPngTool:
    """Tool for converting SVG code to PNG images for visual verification."""
    
    def __init__(self):
        self._temp_dir = tempfile.mkdtemp()
        self._instance_dict = {}
    
    def get_tool_schema(self) -> Dict[str, Any]:
        """Return the tool schema in OpenAI format."""
        return {
            "type": "function",
            "function": {
                "name": "svg_to_png",
                "description": "Converts SVG code to PNG image for visual inspection",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "svg_code": {
                            "type": "string",
                            "description": "Complete SVG code as a string",
                        },
                    },
                    "required": ["svg_code"],
                },
            }
        }
    
    def create_instance(self, instance_id: Optional[str] = None) -> str:
        """Create a new tool instance."""
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "svg_code": "",
            "png_image": None,
        }
        return instance_id
    
    def execute(self, instance_id: str, svg_code: str) -> tuple[Optional[Image.Image], str, bool]:
        """
        Execute the tool to convert SVG to PNG.
        
        Returns:
            tuple: (image, message, success)
        """
        if not isinstance(svg_code, str):
            svg_code = str(svg_code)
        
        self._instance_dict[instance_id]["svg_code"] = svg_code
        
        try:
            # Convert SVG to PNG
            png_path = os.path.join(self._temp_dir, f"{instance_id}.png")
            cairosvg.svg2png(
                bytestring=svg_code.encode('utf-8'),
                write_to=png_path
            )
            
            # Load as PIL Image
            pil_image = Image.open(png_path).convert("RGB")
            processed_image = process_image(pil_image)
            
            self._instance_dict[instance_id]["png_image"] = processed_image
            
            message = (
                "Here is the rendered image from your SVG. "
                "Please inspect it carefully, verify it matches your intent, "
                "and refine the SVG if needed. If it's correct, proceed to solve and finish."
            )
            return processed_image, message, True
            
        except Exception as e:
            error_msg = f"Error converting SVG to PNG: {str(e)}"
            logger.error(error_msg)
            return None, error_msg, False
    
    def release(self, instance_id: str):
        """Release a tool instance."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


# ============================================================================
# Conversation Processing
# ============================================================================


def extract_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Extract tool call from assistant message."""
    start_tag = '<tool_call>'
    end_tag = '</tool_call>'
    
    start = text.find(start_tag)
    if start == -1:
        return None
    
    end = text.find(end_tag, start + len(start_tag))
    if end == -1:
        return None
    
    try:
        payload_str = text[start + len(start_tag):end]
        payload = json.loads(payload_str)
        return payload
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse tool call JSON: {e}")
        return None


def extract_thinking(text: str) -> tuple[str, str]:
    """
    Extract thinking content from text.
    
    Returns:
        tuple: (thinking_content, text_without_thinking)
    """
    start_tag = '<think>'
    end_tag = '</think>'
    
    start = text.find(start_tag)
    
    end = text.find(end_tag, start + len(start_tag))
    if end == -1:
        return "", text
    if start == -1:
        start = -len(start_tag)
    
    thinking = text[start + len(start_tag):end]
    text_without_thinking = text[:start] + text[end + len(end_tag):]
    
    return thinking.strip(), text_without_thinking.strip()


def process_images_from_bytes(images_data: List[Dict[str, Any]]) -> List[Image.Image]:
    """
    Process images from bytes data.
    
    Args:
        images_data: List of image dictionaries with 'bytes' key
    
    Returns:
        List of PIL Images
    """
    processed_images = []
    for img_data in images_data:
        if 'bytes' in img_data:
            from io import BytesIO
            pil_image = Image.open(BytesIO(img_data['bytes'])).convert('RGB')
            processed_images.append(pil_image)
    return processed_images


def attach_images_to_messages(
    messages: List[Dict[str, Any]],
    images: List[Image.Image]
) -> List[Dict[str, Any]]:
    """
    Attach images to messages where <image> placeholders exist.
    
    Args:
        messages: List of messages in HF chat format
        images: List of PIL Images
    
    Returns:
        List of messages with images attached
    """
    result_messages = []
    image_idx = 0
    
    for msg in messages:
        role = msg['role']
        content = msg['content']
        
        # Check if content has <image> placeholder
        if isinstance(content, str) and '<image>' in content:
            # Convert to multi-modal content format
            content_list = []
            remaining_content = content
            
            while '<image>' in remaining_content and image_idx < len(images):
                # Split at <image> placeholder
                before, sep, after = remaining_content.partition('<image>')
                
                if before.strip():
                    content_list.append({"type": "text", "text": before.strip()})
                
                # Add image
                content_list.append({
                    "type": "image",
                    "image": images[image_idx]
                })
                
                remaining_content = after
                image_idx += 1
            
            # Add any remaining text
            if remaining_content.strip():
                content_list.append({"type": "text", "text": remaining_content.strip()})
            
            result_messages.append({"role": role, "content": content_list})
        else:
            # Keep as is
            result_messages.append({"role": role, "content": content})
    
    return result_messages


# ============================================================================
# Multi-turn Inference Engine
# ============================================================================


class MultiTurnInferenceEngine:
    """Multi-turn inference engine with tool use support."""
    
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 8192,
        max_num_seqs: int = 32,
        enable_prefix_caching: bool = True,
    ):
        self.model_path = model_path
        self.tool = SvgToPngTool()
        
        # Initialize tokenizer and processor
        logger.info(f"Loading tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        logger.info(f"Loading processor from {model_path}")
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        
        # Initialize vLLM engine
        logger.info(f"Initializing vLLM engine with model: {model_path}")
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            enable_prefix_caching=enable_prefix_caching,
            limit_mm_per_prompt={"image": 10},  # Support multiple images
        )
        logger.info("vLLM engine initialized successfully")
    
    def generate_single_turn(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 16384,
    ) -> str:
        """Generate a single turn response."""
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Process vision info
        image_inputs, video_inputs = process_vision_info(messages)
        
        # Prepare sampling params
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop_token_ids=[self.tokenizer.eos_token_id],
            extra_args={"enable_thinking": True}
        )
        
        # Generate
        if image_inputs:
            outputs = self.llm.generate(
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": image_inputs},
                },
                sampling_params=sampling_params
            )
        else:
            outputs = self.llm.generate(prompt, sampling_params=sampling_params)
        
        return outputs[0].outputs[0].text
    
    def run_multi_turn_with_tools(
        self,
        initial_messages: List[Dict[str, Any]],
        max_turns: int = 10,
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Run multi-turn inference with tool use.
        
        Returns:
            tuple: (full_conversation, metadata)
        """
        messages = initial_messages.copy()
        instance_id = self.tool.create_instance()
        metadata = {
            "turns": 0,
            "tool_calls": 0,
            "successful_tool_calls": 0,
            "thinking_detected": False,
        }
        
        for turn in range(max_turns):
            # Generate response
            response = self.generate_single_turn(
                messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )
            
            metadata["turns"] += 1
            
            # Extract thinking if present
            thinking, response_without_thinking = extract_thinking(response)
            if thinking:
                metadata["thinking_detected"] = True
            
            # Check for tool call
            tool_call = extract_tool_call(response)
            
            if tool_call:
                metadata["tool_calls"] += 1
                
                # Add assistant message with tool call
                messages.append({"role": "assistant", "content": response})
                
                # Execute tool
                function_name = tool_call.get("name")
                arguments = tool_call.get("arguments", {})
                
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                
                if function_name == "svg_to_png":
                    svg_code = arguments.get("svg_code", "")
                    image, message, success = self.tool.execute(instance_id, svg_code)
                    
                    if success:
                        metadata["successful_tool_calls"] += 1
                        
                        # Add tool response with image
                        tool_message = {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": message}
                            ]
                        }
                        
                        if image:
                            # Save image temporarily
                            temp_image_path = os.path.join(
                                self.tool._temp_dir,
                                f"{instance_id}_result.png"
                            )
                            image.save(temp_image_path)
                            tool_message["content"].append({
                                "type": "image",
                                "image": f"file://{temp_image_path}"
                            })
                        
                        messages.append(tool_message)
                    else:
                        # Tool execution failed
                        messages.append({
                            "role": "user",
                            "content": message
                        })
                else:
                    # Unknown tool
                    messages.append({
                        "role": "user",
                        "content": f"Unknown tool: {function_name}"
                    })
            else:
                # No tool call, this is the final response
                messages.append({"role": "assistant", "content": response})
                break
        
        self.tool.release(instance_id)
        return messages, metadata


# ============================================================================
# Dataset Processing
# ============================================================================


def load_geometry3k_dataset(data_path: str) -> List[Dict[str, Any]]:
    """Load geometry3k dataset from parquet or jsonl file."""
    if data_path.endswith('.parquet'):
        import pandas as pd
        df = pd.read_parquet(data_path)
        data = []
        for idx, row in df.iterrows():
            # Convert numpy arrays to lists
            sample = {}
            for key, value in row.items():
                if hasattr(value, 'tolist'):
                    sample[key] = value.tolist()
                else:
                    sample[key] = value
            data.append(sample)
        return data
    elif data_path.endswith('.jsonl'):
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    else:
        raise ValueError(f"Unsupported file format: {data_path}")


def process_sample(
    sample: Dict[str, Any],
    engine: MultiTurnInferenceEngine,
    image_folder: Optional[str] = None,
    max_turns: int = 10,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_tokens: int = 2048,
) -> Dict[str, Any]:
    """Process a single sample from the dataset."""
    # Support two dataset formats:
    # Format 1: prompt as list of messages, images as list of bytes
    # Format 2: prompt as single string, image as single PIL/bytes object
    
    prompt_messages = sample.get('prompt', [])
    images_data = sample.get('images', [])
    
    # For mathcanvas dataset
    if 'question_images' in sample and 'question_interleave' in sample:
        question_images = sample.get('question_images', [])
        question_interleave = sample.get('question_interleave', [])
        
        # Create messages with system message
        prompt_messages = [{"role": "system", "content": get_system_message()}]
        img_idx = 0
        
        for item in question_interleave:
            if item['type'] == 'text':
                prompt_messages.append({
                    "role": "user",
                    "content": item['content']
                })
            elif item['type'] == 'image':
                if 0 <= img_idx < len(question_images):
                    img_data = question_images[img_idx]
                    images_data.append(img_data)
                    prompt_messages.append({
                        "role": "user",
                        "content": "<image>"
                    })
                    img_idx += 1
        
        # Process images from bytes
        processed_images = process_images_from_bytes(images_data)
    # For ROVER dataset
    elif isinstance(prompt_messages, str) and 'image' in sample:
        # Convert to format 1
        user_message = prompt_messages
        single_image = sample.get('image')
        
        # Create messages with system message and user message with <image> placeholder
        prompt_messages = [
            {"role": "system", "content": get_system_message()},
            {"role": "user", "content": f"<image>\n{user_message}"}
        ]
        
        # Convert single image to images list format
        if isinstance(single_image, Image.Image):
            processed_images = [single_image]
        elif isinstance(single_image, dict) and 'bytes' in single_image:
            images_data = [single_image]
            processed_images = process_images_from_bytes(images_data)
        elif hasattr(single_image, 'read'):  # file-like object
            from io import BytesIO
            pil_image = Image.open(single_image).convert('RGB')
            processed_images = [pil_image]
        else:
            logger.warning("Unsupported image format in 'image' column, skipping")
            return None
    elif not prompt_messages:
        logger.warning("No prompt messages found in sample, skipping")
        return None
    else:
        # Format 1: process images from bytes
        processed_images = process_images_from_bytes(images_data)
    
    # Attach images to messages with <image> placeholders
    initial_messages = attach_images_to_messages(prompt_messages, processed_images)
    
    # Run multi-turn inference
    full_conversation, metadata = engine.run_multi_turn_with_tools(
        initial_messages,
        max_turns=max_turns,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    
    # Create a JSON-serializable version of the sample (exclude image bytes and other non-serializable data)
    def is_json_serializable(obj):
        """Check if an object is JSON serializable."""
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return True
        if isinstance(obj, (list, tuple)):
            return all(is_json_serializable(item) for item in obj)
        if isinstance(obj, dict):
            return all(isinstance(k, str) and is_json_serializable(v) for k, v in obj.items())
        return False
    
    serializable_sample = {}
    for k, v in sample.items():
        if k not in ['images', 'image']:
            if is_json_serializable(v):
                serializable_sample[k] = v
            # Skip non-serializable values
    
    serializable_sample['num_images'] = len(processed_images)
    
    # Convert conversation to serializable format (exclude PIL images)
    serializable_conversation = []
    for msg in full_conversation:
        serializable_msg = {"role": msg["role"]}
        content = msg["content"]
        
        if isinstance(content, str):
            serializable_msg["content"] = content
        elif isinstance(content, list):
            # Filter out image objects, keep only text
            text_content = []
            for item in content:
                if item.get("type") == "text":
                    text_content.append(item)
                # Skip images entirely
            serializable_msg["content"] = text_content
        else:
            serializable_msg["content"] = content
        
        serializable_conversation.append(serializable_msg)
    
    return {
        "original_sample": serializable_sample,
        "generated_conversation": serializable_conversation,
        "metadata": metadata
    }


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Multi-modal, Multi-turn Tool Use Inference with vLLM"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallel size for vLLM"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization for vLLM"
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=8192,
        help="Maximum model sequence length"
    )
    parser.add_argument(
        "--max_num_seqs",
        type=int,
        default=32,
        help="Maximum number of sequences in a batch"
    )
    parser.add_argument(
        "--enable_prefix_caching",
        action="store_true",
        default=True,
        help="Enable prefix caching"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to geometry3k dataset (parquet or jsonl)"
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default=None,
        help="Base folder for image paths (if relative paths in dataset)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save inference results (jsonl)"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to process (None for all)"
    )
    
    # Generation arguments
    parser.add_argument(
        "--max_turns",
        type=int,
        default=10,
        help="Maximum number of turns in conversation"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum tokens per generation"
    )
    
    args = parser.parse_args()
    
    # Load dataset
    logger.info(f"Loading dataset from {args.data_path}")
    dataset = load_geometry3k_dataset(args.data_path)
    
    if args.num_samples:
        dataset = dataset[:args.num_samples]
    
    logger.info(f"Loaded {len(dataset)} samples")
    
    # Initialize engine
    engine = MultiTurnInferenceEngine(
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
    )
    
    # Process samples
    logger.info("Starting inference...")
    results = []
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    with open(args.output_path, 'w') as f:
        for idx, sample in enumerate(tqdm(dataset, desc="Processing samples")):
            result = process_sample(
                sample,
                engine,
                image_folder=args.image_folder,
                max_turns=args.max_turns,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens
            )
            
            print(result)
            if result:
                f.write(json.dumps(result) + '\n')
                f.flush()
                results.append(result)
    
    # Print summary statistics
    total_turns = sum(r["metadata"]["turns"] for r in results)
    total_tool_calls = sum(r["metadata"]["tool_calls"] for r in results)
    successful_tool_calls = sum(r["metadata"]["successful_tool_calls"] for r in results)
    thinking_samples = sum(1 for r in results if r["metadata"]["thinking_detected"])
    
    logger.info("\n" + "="*80)
    logger.info("Inference Summary")
    logger.info("="*80)
    logger.info(f"Total samples processed: {len(results)}")
    logger.info(f"Total turns: {total_turns}")
    logger.info(f"Average turns per sample: {total_turns / len(results):.2f}")
    logger.info(f"Total tool calls: {total_tool_calls}")
    logger.info(f"Successful tool calls: {successful_tool_calls}")
    logger.info(f"Tool call success rate: {successful_tool_calls / total_tool_calls * 100:.2f}%" if total_tool_calls > 0 else "N/A")
    logger.info(f"Samples with thinking tags: {thinking_samples} ({thinking_samples / len(results) * 100:.2f}%)")
    logger.info(f"Results saved to: {args.output_path}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
