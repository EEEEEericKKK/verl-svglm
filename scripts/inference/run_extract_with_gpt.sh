#!/bin/bash
# Script to run extract_results.py with GPT verification using Azure OpenAI credentials

# Azure OpenAI Configuration (from gencot/gpt_filter)
AZURE_OPENAI_KEY="f374165dd51d4f5e9cf61257e1612ecf"
AZURE_OPENAI_ENDPOINT="https://eteopenai.azure-api.net"
AZURE_OPENAI_API_VERSION="2024-08-01-preview"
MODEL_NAME="gpt-5-2025-08-07"

# Check if input file is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <input_jsonl> [output_jsonl] [image_base_dir]"
    echo ""
    echo "Extract answers from inference results with GPT verification"
    echo ""
    echo "Arguments:"
    echo "  input_jsonl    - Path to inference results JSONL file (required)"
    echo "  output_jsonl   - Path to save extracted results (optional)"
    echo "                   Default: input_file with '_extracted.jsonl' suffix"
    echo "  image_base_dir - Base directory for question images (optional)"
    echo "                   If provided, will send images to GPT for verification"
    echo ""
    echo "Example:"
    echo "  $0 ./outputs/inference_results.jsonl"
    echo "  $0 ./outputs/inference_results.jsonl ./outputs/extracted.jsonl"
    echo "  $0 ./outputs/inference_results.jsonl ./outputs/extracted.jsonl /path/to/images"
    exit 1
fi

INPUT_FILE="$1"

# Generate output filename if not provided
if [ -z "$2" ]; then
    # Get directory and basename
    INPUT_DIR=$(dirname "$INPUT_FILE")
    INPUT_BASENAME=$(basename "$INPUT_FILE" .jsonl)
    OUTPUT_FILE="${INPUT_DIR}/${INPUT_BASENAME}_extracted.jsonl"
else
    OUTPUT_FILE="$2"
fi

# Get image base directory (can be overridden by third argument)
if [ -z "$3" ]; then
    IMAGE_BASE_DIR="/proj/inf-scaling/csl/svglm/data/MathCanvas-Bench"
else
    IMAGE_BASE_DIR="$3"
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file not found: $INPUT_FILE"
    exit 1
fi

echo "=========================================="
echo " Extract Results with GPT Verification"
echo "=========================================="
echo "Input:  $INPUT_FILE"
echo "Output: $OUTPUT_FILE"
if [ -n "$IMAGE_BASE_DIR" ]; then
    echo "Images: $IMAGE_BASE_DIR"
fi
echo "Endpoint: $AZURE_OPENAI_ENDPOINT"
echo "Model: $MODEL_NAME"
echo "API Version: $AZURE_OPENAI_API_VERSION"
echo "=========================================="
echo

# Build command
CMD=(python extract_results.py)
CMD+=(--input "$INPUT_FILE")
CMD+=(--output "$OUTPUT_FILE")
CMD+=(--use_gpt_verification)
CMD+=(--api_key "$AZURE_OPENAI_KEY")
CMD+=(--azure_endpoint "$AZURE_OPENAI_ENDPOINT")
CMD+=(--api_version "$AZURE_OPENAI_API_VERSION")
CMD+=(--model "$MODEL_NAME")

# Add image base dir if provided
if [ -n "$IMAGE_BASE_DIR" ]; then
    CMD+=(--image_base_dir "$IMAGE_BASE_DIR")
fi

# Run extraction with GPT verification
"${CMD[@]}"

# Check if extraction was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Extraction completed successfully!"
    echo "Results saved to: $OUTPUT_FILE"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "Extraction failed!"
    echo "=========================================="
    exit 1
fi
