import re
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional

from modules.prompt_improver import PromptImprover
from modules.utils import get_timestamp, append_jsonl
from modules.logger import get_logger

logger = get_logger(__name__)


class LLMOptimizer:
    """Generic LLM API infrastructure: retry logic, logging, response parsing."""

    def __init__(self, api_base: str = "http://localhost:8000/v1",
                 model_name: str = "Qwen/Qwen2.5-VL-72B-Instruct",
                 api_key: str = "dummy", temperature: float = 0.7,
                 use_vision: bool = True, output_dir: str = None):
        self.prompt_improver = PromptImprover(
            api_base=api_base,
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            use_vision=use_vision,
        )
        self.output_dir = Path(output_dir) if output_dir else None
        self._run_timestamp = get_timestamp()

    def _call_api(self, messages: List[Dict], iteration: int) -> Optional[str]:
        """Make an API call with retry. Returns response text or None on failure."""
        max_retries = 3
        for retry in range(max_retries):
            response = requests.post(
                f"{self.prompt_improver.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.prompt_improver.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.prompt_improver.model_name,
                    "messages": messages,
                    "temperature": self.prompt_improver.temperature,
                    "max_tokens": 2048,
                    "top_p": 0.9,
                },
                timeout=180,
            )

            self._save_api_call({
                'iteration': iteration,
                'retry': retry,
                'timestamp': get_timestamp(),
                'input': {
                    'model': self.prompt_improver.model_name,
                    'temperature': self.prompt_improver.temperature,
                    'max_tokens': 2048,
                    'top_p': 0.9,
                },
                'response': {'status_code': response.status_code},
            })

            if response.status_code == 200:
                result = response.json()
                if 'choices' not in result or not result['choices']:
                    print(f"❌ Invalid API response: {result}")
                    continue
                return result['choices'][0]['message']['content']

            elif response.status_code == 429:
                print(f"⚠️ Rate limited, retrying... ({retry + 1}/{max_retries})")
                time.sleep(15 * (retry + 1))

            elif response.status_code >= 500:
                print(f"❌ Server error ({response.status_code}): {response.text[:200]}")
                if retry < max_retries - 1:
                    time.sleep(10)

            else:
                print(f"❌ API error ({response.status_code}): {response.text[:200]}")
                if retry < max_retries - 1:
                    time.sleep(5)

        return None

    @staticmethod
    def _extract_improved_prompt(response: str) -> Optional[str]:
        match = re.search(r'<IMPROVED_PROMPT>(.*?)</IMPROVED_PROMPT>', response,
                          re.DOTALL | re.IGNORECASE)
        if match:
            improved = match.group(1).strip()
            if improved:
                return improved
        return None

    @staticmethod
    def _format_comparison_prompts(prompts: List[str], scores: List[float],
                                   base_score: float, label: str) -> str:
        result = ""
        for i, (prompt, score) in enumerate(zip(prompts, scores)):
            diff = score - base_score
            sign = "+" if diff >= 0 else ""
            result += f"\nExample {i+1} (Score: {score:.4f}, Δ vs Base: {sign}{diff:.4f}):\n{prompt}\n"
        return result

    def _save_api_call(self, record: Dict) -> None:
        if not self.output_dir:
            return
        logs_dir = self.output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        append_jsonl(record, logs_dir / f"api_calls_{self._run_timestamp}.jsonl")
