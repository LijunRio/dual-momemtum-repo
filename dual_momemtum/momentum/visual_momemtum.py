import json
import logging
import base64
import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

VIZ_FONT_SIZE = 28
VIZ_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
]


def _load_viz_font():
    for font_path in VIZ_FONT_PATHS:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, VIZ_FONT_SIZE)
    return ImageFont.load_default()


VIZ_FONT = _load_viz_font()

class VisualMomemtumOptimizer:
    """Visual momentum optimizer: improves prompts by analyzing visual error cases."""

    def __init__(self, api_base: str = "http://localhost:8000/v1",
                 model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct",
                 api_key: str = "dummy",
                 temperature: float = 0.7,
                 output_dir: str = "./visualizations"):
        self.api_base = api_base
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.logger = logging.getLogger('visual_momemtum')

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - VisualMomemtum - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def improve_with_visual_feedback(self, current_prompt: str,
                                     errors: List[Dict],
                                     error_summary: Dict,
                                     iteration,
                                     base_prompt: str = None) -> Optional[str]:
        if not errors:
            self.logger.warning(f"[Iter {iteration}] No error samples, skipping visual optimization")
            return None

        self.logger.info(f"[Iter {iteration}] Starting VisualMomemtum optimization ({len(errors)} errors)")

        representative_errors = self._select_representative_errors(errors, k=3)
        visualizations = self._generate_visualizations(representative_errors, iteration)
        if not visualizations:
            self.logger.warning(f"[Iter {iteration}] Failed to generate visualizations")
            return None

        multi_modal_prompt = self._build_multimodal_prompt(
            current_prompt, representative_errors, visualizations,
            error_summary, iteration, base_prompt=base_prompt
        )

        improved_prompt = self._call_vision_llm(multi_modal_prompt, visualizations, iteration)

        if improved_prompt:
            self.logger.info(f"[Iter {iteration}] ✓ Improved prompt generated")
            self._save_optimization_record(
                iteration=iteration,
                multi_modal_prompt=multi_modal_prompt,
                current_prompt=current_prompt,
                improved_prompt=improved_prompt,
                num_errors=len(errors),
                error_summary=error_summary
            )
        else:
            self.logger.warning(f"[Iter {iteration}] Failed to generate improved prompt")

        return improved_prompt

    def _select_representative_errors(self, errors: List[Dict], k: int = 3) -> List[Dict]:
        if not errors:
            return []

        errors_by_type = {}
        for error in errors:
            errors_by_type.setdefault(error.get('error_type', 'unknown'), []).append(error)

        selected = []
        for error_type in sorted(errors_by_type, key=lambda t: len(errors_by_type[t]), reverse=True):
            if len(selected) >= k:
                break
            candidates = sorted(
                errors_by_type[error_type],
                key=lambda e: (e.get('score', 0.0), e.get('best_iou', 0.0))
            )
            selected.append(candidates[0])

        if len(selected) < k:
            selected_ids = {id(error) for error in selected}
            remaining = [error for error in errors if id(error) not in selected_ids]
            remaining = sorted(
                remaining,
                key=lambda e: (e.get('score', 0.0), e.get('best_iou', 0.0))
            )
            selected.extend(remaining[:k - len(selected)])

        self.logger.info(f"Selected {len(selected)} error samples (k={k}, total={len(errors)}, types={len(errors_by_type)})")
        for i, err in enumerate(selected):
            self.logger.info(
                f"  sample {i+1}: score={err.get('score', 0):.4f}, "
                f"best_iou={err.get('best_iou', 0):.4f}, type={err.get('error_type', 'unknown')}"
            )
        return selected

    def _generate_visualizations(self, errors: List[Dict], iteration) -> List[Dict]:
        visualizations = []
        iter_str = str(iteration)

        for idx, error in enumerate(errors):
            try:
                image_path = error.get('image_path')
                if not image_path or not Path(image_path).exists():
                    self.logger.warning(f"Image path not found: {image_path}")
                    continue

                img = Image.open(image_path).convert('RGB').copy()
                draw = ImageDraw.Draw(img)

                error_type = error.get('error_type', 'unknown')
                score = error.get('score', 0.0)
                pred_boxes = error.get('pred_boxes', [])
                gt_boxes = error.get('gt_boxes', [])
                best_iou = error.get('best_iou', 0.0)

                try:
                    pred_boxes = [[float(c) for c in box] for box in pred_boxes if isinstance(box, list)]
                    gt_boxes = [[float(c) for c in box] for box in gt_boxes if isinstance(box, list)]
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Coordinate conversion failed (error {idx}): {e}")
                    continue

                for box in gt_boxes:
                    try:
                        draw.rectangle([int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                                       outline='green', width=3)
                    except (ValueError, IndexError):
                        continue

                for box in pred_boxes:
                    try:
                        draw.rectangle([int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                                       outline='red', width=3)
                    except (ValueError, IndexError):
                        continue

                try:
                    text = f"Error: {error_type} | Score: {score:.4f} | Best IoU: {best_iou:.3f}"
                    draw.text((10, 10), text, fill='yellow', font=VIZ_FONT)
                except Exception:
                    pass

                viz_path = self.output_dir / f"viz_iter{iter_str}_error{idx:02d}_{error_type}.jpg"
                img.save(viz_path, quality=95)

                visualizations.append({
                    'image_path': str(viz_path),
                    'error_type': error_type,
                    'description': (f"Error Type: {error_type} | mAP@0.5: {score:.4f} | "
                                    f"Best IoU: {best_iou:.4f} | "
                                    f"Pred Boxes: {len(pred_boxes)} | GT Boxes: {len(gt_boxes)}"),
                    'original_image': image_path,
                    'pred_boxes_count': len(pred_boxes),
                    'gt_boxes_count': len(gt_boxes)
                })

            except Exception as e:
                self.logger.error(f"Visualization failed (error {idx}): {e}")
                continue

        return visualizations

    def _build_multimodal_prompt(self, current_prompt: str,
                                 errors: List[Dict],
                                 visualizations: List[Dict],
                                 error_summary: Dict,
                                 iteration,
                                 base_prompt: str = None) -> str:
        error_types_str = ""
        for etype, count in error_summary.get('error_types', {}).items():
            error_types_str += f"  - {etype}: {count} occurrences\n"

        base_prompt_section = f"\n## Base Prompt (Initial Guideline)\n{base_prompt}\n" if base_prompt else ""

        return f"""Your task is to improve a prompt for detecting abnormal areas in medical images.

{base_prompt_section}
## Current Prompt
{current_prompt}

## Error Analysis Context
- Current Iteration: {iteration}
- Total Error Samples: {error_summary.get('total_errors', 0)}
- Error Distribution:
{error_types_str}

## Visual Evidence
I am providing {len(visualizations)} representative error cases below.
In the images:
- **GREEN boxes** = Ground Truth (The correct target)
- **RED boxes** = Model Prediction (Where the model actually looked/detected)

## Task
Analyze the errors above and generate an IMPROVED prompt that:
1. you should first anysis why it make mistakes
2. How to improve it? Tell more precicely what abnomal areas would be or the visual appearcne in the image?
3. Optimize the orignial prompt.
4. You can also change some words in the oringial prompt, like return bouding boxes of any abnormals area, like tumor, white mass, to help the model to more precicely locate the anomaly.
5. Make sure the new prompt should not be too long.

## Output Format
You should output you anaysis process wrapped in <Anaysis> tags
<Anaysis>
[Your Anasysis here]
</Anaysis>

You MUST output the improved prompt wrapped in <IMPROVED_PROMPT> tags:

<IMPROVED_PROMPT>
[Your improved prompt here]
</IMPROVED_PROMPT>
"""

    def _call_vision_llm(self, multi_modal_prompt: str, visualizations: List[Dict], iteration) -> Optional[str]:
        try:
            content = [{"type": "text", "text": multi_modal_prompt}]

            for viz in visualizations:
                try:
                    with open(viz['image_path'], 'rb') as f:
                        image_data = base64.standard_b64encode(f.read()).decode('utf-8')
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}", "detail": "high"}
                    })
                except Exception as e:
                    self.logger.error(f"Failed to load image {viz['image_path']}: {e}")
                    continue

            self.logger.info(f"[Iter {iteration}] Calling Vision LLM...")
            self.logger.info(f"[Iter {iteration}] Input prompt:\n{multi_modal_prompt}")

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": content}],
                temperature=self.temperature,
                max_tokens=2048
            )

            response_text = response.choices[0].message.content
            self.logger.info(f"[Iter {iteration}] LLM response:\n{response_text}")

            improved_prompt = self._extract_prompt(response_text)
            if improved_prompt:
                self.logger.info(f"[Iter {iteration}] ✓ Extracted prompt (len={len(improved_prompt)})")
                return improved_prompt
            else:
                self.logger.warning(f"[Iter {iteration}] Could not parse LLM response")
                return None

        except Exception as e:
            import traceback
            self.logger.error(f"[Iter {iteration}] Vision LLM call failed: {e}\n{traceback.format_exc()}")
            return None

    @staticmethod
    def _extract_prompt(response: str) -> Optional[str]:
        if not response:
            return None

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

        if "```" in response:
            for part in response.split("```")[1::2]:
                content = part.strip()
                if content.startswith(('markdown', 'text', 'json', 'python')):
                    content = '\n'.join(content.split('\n')[1:]).strip()
                if len(content) > 10:
                    return content

        if 50 < len(response) < 5000:
            if "Analysis>" not in response and "Analysis:" not in response[:100]:
                return response.strip()

        return None

    def _save_optimization_record(self, iteration, multi_modal_prompt: str, current_prompt: str,
                                  improved_prompt: str, num_errors: int, error_summary: Dict) -> None:
        record = {
            'timestamp': datetime.now().isoformat(),
            'iteration': iteration,
            'num_errors': num_errors,
            'error_summary': error_summary,
            'input_multimodal_prompt': multi_modal_prompt,
            'current_prompt': current_prompt,
            'improved_prompt': improved_prompt,
        }

        log_dir = self.output_dir / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)

        with open(log_dir / 'visual_momemtum_records.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

        detail_dir = log_dir / 'visual_momemtum_details'
        detail_dir.mkdir(parents=True, exist_ok=True)
        detail_file = detail_dir / f"iteration_{str(iteration).replace('/', '_')}.json"
        with open(detail_file, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        self.logger.info(f"[Iter {iteration}] Optimization record saved to {detail_file}")
