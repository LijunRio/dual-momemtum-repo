import re
from typing import Dict, List, Optional
from io import BytesIO

import requests
import base64

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    ImageDraw = None


class PromptVisualizer:

    def __init__(self, output_dir=None):
        self.output_dir = output_dir
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

    def visualize_error_case(self, image_path, pred_boxes, gt_boxes, error_type, score):
        if not HAS_PIL:
            return None
        try:
            img = Image.open(image_path).convert('RGB')
            draw = ImageDraw.Draw(img, 'RGBA')
            for i, box in enumerate(pred_boxes):
                if len(box) >= 4:
                    x1, y1, x2, y2 = box[:4]
                    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
                    draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 40))
            for i, box in enumerate(gt_boxes):
                if len(box) >= 4:
                    x1, y1, x2, y2 = box[:4]
                    draw.rectangle([x1, y1, x2, y2], outline='green', width=3)
                    draw.rectangle([x1, y1, x2, y2], fill=(0, 255, 0, 40))
            return img
        except Exception as e:
            print(f"Visualization failed: {e}")
            return None

    @staticmethod
    def image_to_base64(pil_image) -> str:
        if pil_image is None:
            return ""
        try:
            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")
            buffer.seek(0)
            return base64.b64encode(buffer.read()).decode()
        except Exception as e:
            print(f"Image encoding failed: {e}")
            return ""


class PromptImprover:

    def __init__(self, api_base: str = "http://localhost:8000/v1",
                 model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct",
                 api_key: str = "dummy", temperature: float = 0.7,
                 use_vision: bool = True):
        self.api_base = api_base
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.use_vision = use_vision

    @staticmethod
    def _extract_prompt(response: str) -> Optional[str]:
        patterns = [
            r'<IMPROVED_PROMPT>(.*?)</IMPROVED_PROMPT>',
            r'<improved_prompt>(.*?)</improved_prompt>',
            r'<PROMPT>(.*?)</PROMPT>',
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                prompt = match.group(1).strip()
                if len(prompt) > 10:
                    return prompt
        return None
