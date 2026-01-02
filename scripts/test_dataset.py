from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from transformers import AutoTokenizer, AutoProcessor
from omegaconf import OmegaConf

# Load your dataset
tokenizer = AutoTokenizer.from_pretrained("/proj/inf-scaling/csl/svglm/checkpoints/Qwen3-VL-8B-Thinking")
processor = AutoProcessor.from_pretrained("/proj/inf-scaling/csl/svglm/checkpoints/Qwen3-VL-8B-Thinking")

config = OmegaConf.create({
    "max_length": 2048,
    "truncation": "error",
    "messages_key": "messages",
    "image_key": "images",
    "tools_key": "tools",
    "apply_chat_template_kwargs": {}
})

dataset = MultiTurnSFTDataset(
    parquet_files="/proj/inf-scaling/csl/svglm/data/geo3k_toolcall/processed_data_verl.parquet",
    tokenizer=tokenizer,
    processor=processor,
    config=config
)

# Inspect a sample
sample = dataset[0]

# Get raw data from dataframe
import json
raw_data = dataset.dataframe.iloc[0].to_dict()

# Open output file
with open('output.txt', 'w', encoding='utf-8') as f:
    # Write raw data first
    f.write("=" * 80 + "\n")
    f.write("RAW DATA FROM PARQUET (before tokenization)\n")
    f.write("=" * 80 + "\n\n")
    
    # Pretty print the raw data
    for key, value in raw_data.items():
        f.write(f"{key}:\n")
        if isinstance(value, (list, dict)):
            f.write(json.dumps(value, indent=2, ensure_ascii=False))
        else:
            f.write(str(value))
        f.write("\n\n")
    
    f.write("\n")
    
    # Continue with processed sample data
    f.write("=" * 80 + "\n")
    f.write("DATASET SAMPLE 0 INSPECTION\n")
    f.write("=" * 80 + "\n\n")
    
    # Write shapes
    f.write(f"Input IDs shape: {sample['input_ids'].shape}\n")
    f.write(f"Loss mask shape: {sample['loss_mask'].shape}\n")
    f.write(f"Attention mask shape: {sample['attention_mask'].shape}\n")
    f.write(f"Position IDs shape: {sample['position_ids'].shape}\n")
    
    if 'multi_modal_inputs' in sample:
        f.write(f"\nMulti-modal inputs keys: {list(sample['multi_modal_inputs'].keys())}\n")
        for key, value in sample['multi_modal_inputs'].items():
            f.write(f"  {key}: {value.shape}\n")
    
    # Statistics
    total_tokens = len(sample['input_ids'])
    trained_tokens = sample['loss_mask'].sum().item()
    f.write(f"\nTotal tokens: {total_tokens}\n")
    f.write(f"Tokens to train (loss_mask == 1): {trained_tokens}\n")
    f.write(f"Training ratio: {trained_tokens/total_tokens*100:.2f}%\n")
    
    # Decode to see the actual text (remove image_pad tokens for readability)
    full_text = tokenizer.decode(sample['input_ids'])
    full_text_clean = full_text.replace('<|image_pad|>', '[IMAGE_TOKEN]')
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("FULL TOKENIZED TEXT (cleaned for readability)\n")
    f.write("=" * 80 + "\n")
    f.write(full_text_clean)
    
    # See only the parts that will be trained (where loss_mask == 1)
    assistant_text = tokenizer.decode(sample['input_ids'][sample['loss_mask'] == 1])
    f.write("\n\n" + "=" * 80 + "\n")
    f.write("ASSISTANT RESPONSES (trained parts only)\n")
    f.write("=" * 80 + "\n")
    f.write(assistant_text)
    f.write("\n")

print("Sample saved to output.txt")