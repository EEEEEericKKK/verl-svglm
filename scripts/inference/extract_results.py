#!/usr/bin/env python
"""
Extract ground truth answer, model final answer, and successful tool calls 
from vLLM multi-turn inference results.
"""

import argparse
import json
import re
from pathlib import Path


def extract_final_answer(text: str) -> str:
    """
    Extract text after 'Final answer:' from the model output.
    
    Args:
        text: Full model response text
        
    Returns:
        Extracted answer string or empty string if not found
    """
    # Look for "Final answer:" pattern (case insensitive)
    pattern = r'[Ff]inal\s+[Aa]nswer:\s*(.+?)(?:\n|$|</answer>)'
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    return ""


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
                # Check if this message contains an answer block
                if '<answer>' in content:
                    # Extract answer from within <answer> tags
                    answer_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                    if answer_match:
                        answer_text = answer_match.group(1).strip()
                        # Now extract final answer from within the answer block
                        final_answer = extract_final_answer(answer_text)
                        if final_answer:
                            return final_answer
                        # If no "Final answer:" found, return the whole answer block
                        return answer_text
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
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    extracted_count = 0
    total_count = 0
    
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line_num, line in enumerate(infile, 1):
            total_count += 1
            
            try:
                data = json.loads(line)
                
                # Extract ground truth answer
                ground_truth = data.get('original_sample', {}).get('extra_info', {}).get('answer', '')
                if ground_truth == '':
                    ground_truth = data.get('original_sample', {}).get('answer', '')
                
                # Extract model's final answer from conversation
                conversation = data.get('generated_conversation', [])
                model_answer = extract_answer_from_conversation(conversation)
                
                # Extract successful tool calls
                successful_tool_calls = data.get('metadata', {}).get('successful_tool_calls', 0)
                
                # Create extracted record
                extracted_record = {
                    'index': data.get('original_sample', {}).get('extra_info', {}).get('index', line_num - 1),
                    'question': data.get('original_sample', {}).get('extra_info', {}).get('question', ''),
                    'ground_truth': ground_truth,
                    'model_answer': model_answer,
                    'successful_tool_calls': successful_tool_calls,
                    'total_turns': data.get('metadata', {}).get('turns', 0),
                    'total_tool_calls': data.get('metadata', {}).get('tool_calls', 0)
                }
                
                outfile.write(json.dumps(extracted_record) + '\n')
                extracted_count += 1
                
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line {line_num}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}")
                continue
    
    print(f"Extraction complete!")
    print(f"Total records processed: {total_count}")
    print(f"Successfully extracted: {extracted_count}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
