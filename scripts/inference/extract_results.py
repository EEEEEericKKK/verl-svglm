#!/usr/bin/env python
"""
Extract ground truth answer, model final answer, and successful tool calls 
from vLLM multi-turn inference results.
"""

import argparse
import base64
import json
import logging
import os
import re
from io import BytesIO
from pathlib import Path
from typing import Optional, List

from openai import AzureOpenAI
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')


def last_boxed_only_string(string: str) -> Optional[str]:
    """Extract the last LaTeX boxed expression from a string.

    Args:
        string: Input string containing LaTeX code

    Returns:
        The last boxed expression or None if not found
    """
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0

    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def remove_boxed(s: str) -> str:
    """Remove the LaTeX boxed command from a string.

    Args:
        s: String with format "\\boxed{content}"

    Returns:
        The content inside the boxed command
    """
    left = "\\boxed{"
    assert s[: len(left)] == left, f"box error: {s}"
    assert s[-1] == "}", f"box error: {s}"
    return s[len(left) : -1]


def verify_answer_with_gpt(
    question: str,
    ground_truth: str,
    model_answer: str,
    azure_client: AzureOpenAI,
    model_id: str,
    question_images: Optional[List[Image.Image]] = None,
    max_retries: int = 3
) -> Optional[dict]:
    """
    Use GPT to verify if the model's answer is correct.
    
    Args:
        question: The original question
        ground_truth: The ground truth answer
        model_answer: The model's predicted answer
        azure_client: AzureOpenAI client instance
        model_id: Model deployment name
        question_images: Optional list of PIL Images for the question
        max_retries: Maximum number of retry attempts
    
    Returns:
        Dictionary with verification results or None if failed
    """
    # Build the content list with text and images
    content = [
        {
            "type": "text",
            "text": f"""You are a math answer verification assistant. Your task is to determine if a model's answer is correct compared to the ground truth.

Question: {question}

Ground Truth Answer: {ground_truth}

Model's Answer: {model_answer}

IMPORTANT: The question may contain multiple sub-questions. The ground truth and model answer may be provided as lists or separate values for each sub-question.

Please evaluate whether the model's answer is mathematically equivalent to the ground truth answer for EACH sub-question. Consider:
- Numerical equivalence (e.g., 0.5 = 1/2)
- Algebraic equivalence (e.g., simplified vs unsimplified forms)
- Minor formatting differences
- Different representations of the same value
- If a statement is given instead of a choice, check if it complies with the question and image

Respond in JSON format:
{{
    "evaluations": [
        {{
            "sub_question_index": 0,
            "is_correct": true/false,
            "reason": "Brief explanation of your decision",
            "confidence": "high/medium/low"
        }},
        ...
    ],
    "overall_correct": true/false,
    "num_sub_questions": <number>
}}

Notes:
- If there is only one question, provide one evaluation in the list
- overall_correct should be true only if ALL sub-questions are correct
- num_sub_questions should be the total number of sub-questions identified"""
        }
    ]
    
    # Add question images if provided
    if question_images:
        for idx, img in enumerate(question_images):
            content.append({
                "type": "text",
                "text": f"Question Image {idx + 1}:"
            })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{pil_to_base64(img)}"
                }
            })

    for attempt in range(max_retries):
        try:
            response = azure_client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
            )
            
            result_text = response.choices[0].message.content
            
            # Try to parse as JSON
            try:
                result_json = json.loads(result_text)
                return result_json
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code block
                json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', result_text, re.DOTALL)
                if json_match:
                    result_json = json.loads(json_match.group(1))
                    return result_json
                # If not valid JSON, return as text
                logger.warning(f"Could not parse GPT response as JSON: {result_text}")
                return {"raw_response": result_text}
            
        except Exception as e:
            logger.warning(f"GPT verification attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                logger.error(f"All retries failed for answer verification")
                return None
    
    return None


def extract_final_answer(text: str) -> str:
    """
    Extract the final answer using \\boxed{} format.
    
    Args:
        text: Full model response text
        
    Returns:
        Extracted answer string or empty string if not found
    """
    # First try to extract from \boxed{}
    boxed_answer = last_boxed_only_string(text)
    if boxed_answer:
        try:
            return remove_boxed(boxed_answer)
        except (AssertionError, IndexError):
            # If boxed extraction fails, return the whole boxed string
            return boxed_answer
    
    return ""


def load_question_images(question_interleave: list, image_base_dir: Optional[str]) -> List[Image.Image]:
    """
    Load question images from interleave format.
    
    Args:
        question_interleave: List of interleaved text and image items
        image_base_dir: Base directory for image files
    
    Returns:
        List of PIL Images
    """
    images = []
    if not question_interleave or not image_base_dir:
        return images
    
    for item in question_interleave:
        if item.get('type') == 'image':
            img_path = item.get('content', '')
            if img_path:
                # Construct full path
                full_path = os.path.join(image_base_dir, img_path)
                try:
                    img = Image.open(full_path).convert('RGB')
                    images.append(img)
                except Exception as e:
                    logger.warning(f"Failed to load image {full_path}: {e}")
    
    return images


def extract_image_paths(question_interleave: list) -> List[str]:
    """
    Extract image paths from interleave format.
    
    Args:
        question_interleave: List of interleaved text and image items
    
    Returns:
        List of image paths
    """
    image_paths = []
    if not question_interleave:
        return image_paths
    
    for item in question_interleave:
        if item.get('type') == 'image':
            img_path = item.get('content', '')
            if img_path:
                image_paths.append(img_path)
    
    return image_paths


def extract_answer_from_conversation(conversation: list) -> str:
    """
    Extract the final answer from the last assistant message in conversation.
    
    Args:
        conversation: List of conversation messages
        
    Returns:
        Extracted final answer
    """
    # Find the last assistant message
    for msg in reversed(conversation):
        if msg.get('role') == 'assistant':
            content = msg.get('content', '')
            if isinstance(content, str):
                # Check if this message contains an answer with \boxed{}
                if '\\boxed' in content:
                    # Use proper bracket matching to extract boxed content
                    boxed_answer = last_boxed_only_string(content)
                    if boxed_answer:
                        try:
                            answer_text = remove_boxed(boxed_answer)
                            return answer_text.strip()
                        except (AssertionError, IndexError):
                            # If remove_boxed fails, return the whole boxed string
                            return boxed_answer
    return ""


def main():
    parser = argparse.ArgumentParser(
        description="Extract answers and metadata from inference results"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input JSONL file with inference results"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSONL file with extracted data"
    )
    parser.add_argument(
        "--use_gpt_verification",
        action="store_true",
        help="Use GPT to verify answer correctness"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="Azure OpenAI API key (required if --use_gpt_verification)"
    )
    parser.add_argument(
        "--azure_endpoint",
        type=str,
        default=None,
        help="Azure OpenAI endpoint (required if --use_gpt_verification)"
    )
    parser.add_argument(
        "--api_version",
        type=str,
        default="2024-02-15-preview",
        help="Azure OpenAI API version"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5",
        help="Model deployment name for verification"
    )
    parser.add_argument(
        "--image_base_dir",
        type=str,
        default=None,
        help="Base directory for question images (required if using GPT verification with images)"
    )    
    args = parser.parse_args()
    
    # Initialize Azure OpenAI client if needed
    azure_client = None
    if args.use_gpt_verification:
        if not args.api_key or not args.azure_endpoint:
            raise ValueError("--api_key and --azure_endpoint are required when using GPT verification")
        azure_client = AzureOpenAI(
            api_key=args.api_key,
            azure_endpoint=args.azure_endpoint,
            api_version=args.api_version,
        )
        logger.info(f"GPT verification enabled using model: {args.model}")
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    extracted_count = 0
    total_count = 0
    gpt_verified_count = 0
    gpt_correct_count = 0
    
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            total_count += 1
            
            try:
                data = json.loads(line)
                
                original_sample = data.get('original_sample', {})
                
                # Extract sample ID
                sample_id = original_sample.get('id', f'sample_{line_num}')
                
                # Extract ground truth answer (now at root level after refactor)
                ground_truth = data.get('ground_truth', '')
                if not ground_truth:
                    # Fallback to original_sample for backward compatibility
                    ground_truth = original_sample.get('answer', '')
                
                # Extract question from interleave format
                question_interleave = original_sample.get('question_interleave', [])
                
                # Parse if it's a string representation of list
                if isinstance(question_interleave, str):
                    import ast
                    try:
                        question_interleave = ast.literal_eval(question_interleave)
                    except:
                        logger.warning(f"Sample {sample_id}: Failed to parse question_interleave")
                        question_interleave = []
                
                question_parts = []
                for item in question_interleave:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        question_parts.append(item.get('content', ''))
                question = ' '.join(question_parts).strip()
                
                # Fallback if no interleave format found
                if not question:
                    question = original_sample.get('question', '')
                
                # Extract model's final answer from conversation
                conversation = data.get('generated_conversation', [])
                model_answer = extract_answer_from_conversation(conversation)
                
                # Extract image paths if available
                image_paths = extract_image_paths(question_interleave) if question_interleave else []
                
                # Extract successful tool calls
                successful_tool_calls = data.get('metadata', {}).get('successful_tool_calls', 0)
                
                # Create extracted record
                extracted_record = {
                    'sample_id': sample_id,
                    'question': question,
                    'ground_truth': ground_truth,
                    'model_answer': model_answer,
                    'image_paths': image_paths,
                    'successful_tool_calls': successful_tool_calls,
                    'total_turns': data.get('metadata', {}).get('turns', 0),
                    'total_tool_calls': data.get('metadata', {}).get('tool_calls', 0),
                    'knowledge': original_sample.get('knowledge', ''),
                    'subknowledge': original_sample.get('subknowledge', '')
                }
                
                # GPT verification if enabled
                if args.use_gpt_verification and azure_client and question and ground_truth and model_answer:
                    # Load question images if available
                    question_images = None
                    if args.image_base_dir and question_interleave:
                        question_images = load_question_images(question_interleave, args.image_base_dir)
                    
                    logger.info(f"Processing sample {line_num}/{total_count} (ID: {sample_id}) - Verifying with GPT...")
                    
                    gpt_result = verify_answer_with_gpt(
                        question=question,
                        ground_truth=ground_truth,
                        model_answer=model_answer,
                        azure_client=azure_client,
                        model_id=args.model,
                        question_images=question_images
                    )
                    
                    if gpt_result:
                        gpt_verified_count += 1
                        extracted_record['gpt_verification'] = gpt_result
                        # Check overall_correct field (or fallback to is_correct for backward compatibility)
                        is_correct = gpt_result.get('overall_correct', gpt_result.get('is_correct', False))
                        if is_correct:
                            gpt_correct_count += 1
                        logger.info(f"  ✓ Sample {sample_id}: {'CORRECT' if is_correct else 'INCORRECT'} (Verified: {gpt_verified_count}, Correct: {gpt_correct_count})")
                    else:
                        extracted_record['gpt_verification'] = {'error': 'verification_failed'}
                        logger.warning(f"  ✗ Sample {sample_id}: Verification failed")
                else:
                    if line_num % 10 == 0:
                        logger.info(f"Processing sample {line_num}/{total_count} (ID: {sample_id})")
                
                outfile.write(json.dumps(extracted_record) + '\n')
                outfile.flush()  # Flush to disk immediately
                extracted_count += 1
                
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Error processing line {line_num}: {e}")
                continue
    
    logger.info(f"Extraction complete!")
    logger.info(f"Total records processed: {total_count}")
    logger.info(f"Successfully extracted: {extracted_count}")
    if args.use_gpt_verification:
        logger.info(f"GPT verified: {gpt_verified_count}")
        logger.info(f"GPT marked correct: {gpt_correct_count} ({gpt_correct_count/gpt_verified_count*100:.1f}%)" if gpt_verified_count > 0 else "GPT marked correct: 0")
    logger.info(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
