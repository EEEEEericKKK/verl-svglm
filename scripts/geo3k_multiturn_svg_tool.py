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
Preprocess the Geometry3k dataset to parquet format
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir",
        default="/proj/inf-scaling/csl/svglm/data/geo3k_multiturn_rl",
        help="The save directory for the preprocessed dataset.",
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "hiyouga/geometry3k"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source)

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # System prompt from agent.py
    system_prompt = (
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

    # Instruction following prompt from agent.py  
    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. "
        r"The final answer MUST BE put in \boxed{}."
    )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("problem")
            # prompt = problem + " " + instruction_following
            prompt = problem
            answer = example.pop("answer")
            images = example.pop("images")
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "images": images,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "svg_to_png": {
                            "create_kwargs": {"dummy": None},
                            # "execute_kwargs": {},
                            # "calc_reward_kwargs": {},
                            # "release_kwargs": {},
                        },
                    },
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_save_dir, dst=hdfs_dir)
