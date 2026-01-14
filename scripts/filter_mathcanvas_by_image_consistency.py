# Copyright 2023-2025 SGLang Team
# Copyright Amazon.com, Inc. or its affiliates.
# Copyright 2025 Reallm Labs Ltd. or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Filter MathCanvas-Instruct dataset by image consistency
Only keep entries with:
1. Exactly one image in question_images and one in solution_images
2. At least 80% pixel-level consistency between the two images
"""

import argparse
import os
from io import BytesIO
from multiprocessing import Pool

import datasets
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_image_from_path_or_bytes(img):
    """Load PIL Image from path or bytes"""
    if isinstance(img, str):
        return Image.open(img)
    elif isinstance(img, bytes):
        return Image.open(BytesIO(img))
    elif isinstance(img, Image.Image):
        return img
    else:
        raise ValueError(f"Unsupported image type: {type(img)}")


def check_image_consistency(img1, img2, threshold=0.8):
    """
    Check if two images have at least threshold pixel-level consistency.
    
    Args:
        img1: First image (PIL Image, path, or bytes)
        img2: Second image (PIL Image, path, or bytes)
        threshold: Minimum consistency ratio (default 0.8 for 80%)
    
    Returns:
        bool: True if consistency >= threshold, False otherwise
    """
    try:
        # Load images
        image1 = load_image_from_path_or_bytes(img1)
        image2 = load_image_from_path_or_bytes(img2)
        
        # Convert to RGB if needed
        if image1.mode != 'RGB':
            image1 = image1.convert('RGB')
        if image2.mode != 'RGB':
            image2 = image2.convert('RGB')
        
        # Check aspect ratios
        aspect_ratio1 = image1.width / image1.height
        aspect_ratio2 = image2.width / image2.height
        
        # If aspect ratios differ by more than 1%, filter out
        if abs(aspect_ratio1 - aspect_ratio2) / aspect_ratio1 > 0.01:
            return False
        
        # Resize both images to the same size (use the smaller dimensions)
        target_size = (
            min(image1.width, image2.width),
            min(image1.height, image2.height)
        )
        
        image1_resized = image1.resize(target_size, Image.Resampling.LANCZOS)
        image2_resized = image2.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy arrays
        arr1 = np.array(image1_resized)
        arr2 = np.array(image2_resized)

        # Create mask for non-white pixels (ignore white or near-white pixels)
        # Consider pixels as "white" if all RGB channels are >= 250
        non_white_mask1 = np.any(arr1 < 250, axis=-1)
        non_white_mask2 = np.any(arr2 < 250, axis=-1)
        # Combine masks - pixel must be non-white in at least one image
        non_white_mask = non_white_mask1 | non_white_mask2
        
        # If all pixels are white, skip comparison
        if not np.any(non_white_mask):
            return False
        
        # Calculate pixel-level consistency only for non-white pixels
        arr1_masked = arr1[non_white_mask]
        arr2_masked = arr2[non_white_mask]
        diff = np.sum(np.abs(arr1_masked - arr2_masked), axis=-1)
        
        total_pixels = diff.size
        matching_pixels = np.sum(diff < 20)
        consistency = matching_pixels / total_pixels
        return consistency >= threshold
        
    except Exception as e:
        print(f"Error checking image consistency: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="/proj/inf-scaling/csl/svglm/data/mathcanvas_filtered",
        help="The save directory for the filtered dataset.",
    )
    parser.add_argument(
        "--consistency_threshold",
        type=float,
        default=0.5,
        help="Minimum pixel consistency threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )
    
    args = parser.parse_args()
    
    data_source = "shiwk24/MathCanvas-Bench"
    subsets = ["Solid_Geometry", "Plane_Geometry", "Trigonometry"]
    # subsets = ["Plane_Geometry"]
    
    print(f"Loading datasets from {data_source}...")
    print(f"Subsets: {subsets}")
    
    # Load datasets
    dataset_splits = [datasets.load_dataset(data_source, subset) for subset in subsets]
    
    # Concatenate all training splits
    full_dataset = datasets.concatenate_datasets(
        [ds["test"] for ds in dataset_splits]
    )
    
    print(f"Total samples before filtering: {len(full_dataset)}")
    
    # Limit samples for testing if specified
    if args.max_samples is not None:
        print(f"Limiting to {args.max_samples} samples for testing...")
        full_dataset = full_dataset.select(range(min(args.max_samples, len(full_dataset))))
    
    # Apply filtering
    print(f"Filtering by image criteria (consistency threshold: {args.consistency_threshold})...")
    print(f"Using 8 worker processes for parallel processing...")
    
    # Create a custom filter function with the threshold
    def filter_fn(idx):
        example = full_dataset[idx]
        question_images = example.get("question_images", [])
        solution_images = example.get("solution_images", [])
        
        # Check if both have exactly one image
        if len(question_images) != 1 or len(solution_images) != 1:
            return None
        
        # Check pixel-level consistency
        if check_image_consistency(
            question_images[0], 
            solution_images[0], 
            threshold=args.consistency_threshold
        ):
            return idx
        return None
    
    # Use multiprocessing with progress bar
    with Pool(processes=8) as pool:
        results = list(tqdm(
            pool.imap(filter_fn, range(len(full_dataset))),
            total=len(full_dataset),
            desc="Filtering samples"
        ))
    
    # Collect valid indices (filter out None values)
    filtered_indices = [idx for idx in results if idx is not None]
    
    filtered_dataset = full_dataset.select(filtered_indices)
    
    print(f"Total samples after filtering: {len(filtered_dataset)}")
    print(f"Retention rate: {len(filtered_dataset) / len(full_dataset) * 100:.2f}%")
    
    # Save filtered dataset
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "filtered_dataset.parquet")
    
    print(f"Saving filtered dataset to {output_path}...")
    filtered_dataset.to_parquet(output_path)
    
    # Visualize a small portion of the filtered dataset
    print("\nCreating visualization of filtered samples...")
    num_samples_to_visualize = min(5, len(filtered_dataset))
    
    if num_samples_to_visualize > 0:
        fig, axes = plt.subplots(num_samples_to_visualize, 2, figsize=(12, 4 * num_samples_to_visualize))
        
        # Handle case where there's only one sample
        if num_samples_to_visualize == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples_to_visualize):
            sample = filtered_dataset[i]
            question_img = load_image_from_path_or_bytes(sample["question_images"][0])
            solution_img = load_image_from_path_or_bytes(sample["solution_images"][0])
            
            # Display question image
            axes[i, 0].imshow(question_img)
            axes[i, 0].set_title(f"Sample {i+1}: Question Image\nID: {sample.get('id', 'N/A')}")
            axes[i, 0].axis('off')
            
            # Display solution image
            axes[i, 1].imshow(solution_img)
            axes[i, 1].set_title(f"Sample {i+1}: Solution Image")
            axes[i, 1].axis('off')
        
        plt.tight_layout()
        viz_path = os.path.join(args.output_dir, "filtered_samples_visualization.png")
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {viz_path}")
        plt.close()
    else:
        print("No samples to visualize.")
    
    print("Done!")
