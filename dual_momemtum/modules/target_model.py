import os
import traceback
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from modules.evluate import parse_boxes_from_output
from modules.logger import get_logger

logger = get_logger(__name__)


class TargetModelRunner:

    def __init__(self, model_path: str, gpu_memory_utilization: float = 0.9,
                 batch_size: int = 64, tensor_parallel_size: Optional[int] = None,
                 image_path_replacement: Optional[Dict[str, str]] = None):
        self.model_path = model_path
        self.batch_size = batch_size
        self.image_path_replacement = image_path_replacement or {"from": "", "to": ""}
        self.tensor_parallel_size = (torch.cuda.device_count()
                                     if tensor_parallel_size is None
                                     else max(1, tensor_parallel_size))
        self._llm = None
        self._processor = None
        self._gen_params = None
        self.gpu_memory_utilization = gpu_memory_utilization

    def init_model(self, temperature: float = 0.1, top_p: float = 1.0,
                   max_gen_tokens: int = 512, max_tokens: int = 8192, seed: int = 42):
        if self._llm is not None:
            return
        print(f"Loading target model (GPUs={self.tensor_parallel_size}, "
              f"mem_util={self.gpu_memory_utilization})...")
        try:
            self._llm = LLM(
                model=self.model_path,
                gpu_memory_utilization=self.gpu_memory_utilization,
                tensor_parallel_size=self.tensor_parallel_size,
                pipeline_parallel_size=1,
                max_model_len=max_tokens,
                dtype="bfloat16",
                enforce_eager=True,
                limit_mm_per_prompt={"image": 1, "video": 1},
            )
            self._processor = AutoProcessor.from_pretrained(self.model_path)
            self._gen_params = SamplingParams(
                temperature=0.0, top_p=1, top_k=-1, n=1, max_tokens=max_gen_tokens
            )
            print("✅ Target model loaded")
        except Exception as e:
            print(f"Model load failed: {e}")
            traceback.print_exc()
            raise

    def fix_image_path(self, image_path: str) -> str:
        if self.image_path_replacement["from"] in image_path:
            return image_path.replace(
                self.image_path_replacement["from"],
                self.image_path_replacement["to"]
            )
        return image_path

    def extract_sample_info(self, sample: Dict) -> Tuple[Optional[str], Optional[str]]:
        try:
            images = sample.get('images', [])
            if not images:
                return None, None

            question = None
            for msg in sample.get('messages', []):
                if msg.get('role') == 'user':
                    content = msg.get('content', [])
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'text':
                                question = item.get('text')
                                break
                    elif isinstance(content, str):
                        question = content
                    if question:
                        break

            if not question:
                question = "Detect abnormal areas in this medical image."

            if isinstance(images, str):
                image_path = images
            elif isinstance(images, list):
                image_path = images[0] if images else None
            else:
                return None, None

            if image_path is None:
                return None, None

            return question, self.fix_image_path(image_path)
        except Exception as e:
            print(f"Failed to extract sample info: {e}")
            return None, None

    def prepare_multimodal_batch(self, questions: list, image_paths: list):
        if self._processor is None:
            raise RuntimeError("Processor not initialized")

        from qwen_vl_utils import process_vision_info

        prompts = []
        mm_data_list = []

        for question, image_path in zip(questions, image_paths):
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": question},
                ],
            }]
            prompt = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)
            image_inputs, _ = process_vision_info(messages)
            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            mm_data_list.append(mm_data)

        return prompts, mm_data_list

    def infer_batch(self, mini_batch: List[Dict], prompt: str) -> List[Dict]:
        batch_questions = []
        batch_image_paths = []
        batch_valid_indices = []
        batch_samples = []

        for idx, sample in enumerate(mini_batch):
            question, image_path = self.extract_sample_info(sample)
            if question is None or image_path is None:
                continue
            if not os.path.exists(image_path):
                continue
            batch_questions.append(prompt + "\n" + question)
            batch_image_paths.append(image_path)
            batch_valid_indices.append(idx)
            batch_samples.append(sample)

        if not batch_questions:
            return []

        try:
            prompts, mm_data_list = self.prepare_multimodal_batch(batch_questions, batch_image_paths)
            llm_inputs = [
                {"prompt": p, "multi_modal_data": mm}
                for p, mm in zip(prompts, mm_data_list)
            ]
            outputs_batch = self._llm.generate(llm_inputs, sampling_params=self._gen_params)
            answers = [o.outputs[0].text.strip() for o in outputs_batch]

            results = []
            for idx, answer in enumerate(answers):
                result = {
                    'sample_idx': batch_valid_indices[idx],
                    'image_path': batch_image_paths[idx],
                    'sample': batch_samples[idx],
                    'answer': answer,
                    'parsed_boxes': parse_boxes_from_output(answer),
                }
                results.append(result)
                logger.debug(f"sample {result['sample_idx']}: {batch_image_paths[idx]}")
                logger.debug(f"  output: {answer[:150]}...")
                logger.debug(f"  parsed: {result['parsed_boxes']}")

            return results

        except Exception as e:
            print(f"Inference error: {e}")
            traceback.print_exc()
            return []
