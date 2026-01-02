# Copyright 2023-2025 SGLang Team
# Copyright Amazon.com, Inc. or its affiliates.
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

import logging
import os
import tempfile
from typing import Any, Optional
from uuid import uuid4

import cairosvg
from PIL import Image

from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.dataset.vision_utils import process_image

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class SvgToPngTool(BaseTool):
    """A tool for converting SVG code to PNG images for visual verification.
    - `get_openai_tool_schema`: return the tool schema in OpenAI format.
    - `create`: create a tool instance for a trajectory.
    - `execute`: execute the tool.
    - `calc_reward`: calculate the reward respect to tool state.
    - `release`: release the tool instance.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
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
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self._temp_dir = tempfile.mkdtemp()

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self, instance_id: Optional[str] = None, **kwargs
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "svg_code": "",
            "png_image": None,
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        svg_code = parameters.get("svg_code", "")
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
            
            # Load as PIL Image and process
            pil_image = Image.open(png_path).convert("RGB")
            processed_image = process_image(pil_image)
            
            self._instance_dict[instance_id]["png_image"] = processed_image
            
            # Tool response with visual feedback message from agent.py
            tool_response = ToolResponse(
                text=(
                    "Here is the rendered image from your SVG. "
                    "Please inspect it carefully, verify it matches your intent, "
                    "and refine the SVG if needed. If it's correct, proceed to solve and finish."
                ),
                image=[processed_image]
            )
            return tool_response, 0.0, {}
            
        except Exception as e:
            error_msg = f"Error converting SVG to PNG: {str(e)}"
            logger.error(error_msg)
            return ToolResponse(text=error_msg), -0.1, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        # SVG to PNG conversion doesn't have a reward - just returns 0
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
