import json
import re
from pathlib import Path
from typing import Dict
from datetime import datetime

from modules.utils import ensure_dir, read_text_file, write_text_file, save_json
from modules.prompt_improver import PromptImprover


class PromptManager:

    def __init__(self, instructions_dir: str, output_dir: str):
        self.instructions_dir = Path(instructions_dir)
        self.output_dir = ensure_dir(Path(output_dir))
        self.prompts = {}
        self.scores = {}
        self.history = []

    def load_base_prompt(self) -> str:
        base_file = self.instructions_dir / "base.txt"
        if not base_file.exists():
            raise FileNotFoundError(f"Cannot find base.txt: {base_file}")
        prompt = read_text_file(base_file)
        self.prompts['base'] = prompt
        print(f"✓ Loaded base prompt: {len(prompt)} characters")
        return prompt

    def load_variant_prompts(self) -> Dict[str, str]:
        variants = {}
        variants_dir = self.instructions_dir / "varients"
        if not variants_dir.exists():
            variants_dir.mkdir(parents=True, exist_ok=True)

        for file in variants_dir.glob("*.txt"):
            try:
                prompt = read_text_file(file)
                name = file.stem
                variants[name] = prompt
                self.prompts[name] = prompt
                print(f"✓ Loaded variant '{name}': {len(prompt)} characters")
            except Exception as e:
                print(f"⚠️ Failed to load {file.name}: {e}")

        return variants

    def generate_initial_variants_if_needed(self, base_prompt: str, llm_config: Dict) -> Dict[str, str]:
        """Generate initial variants via LLM if instructions/varients/ is empty."""
        variants_dir = self.instructions_dir / "varients"
        existing = list(variants_dir.glob("*.txt")) if variants_dir.exists() else []
        if existing:
            print(f"✓ Variants directory already has {len(existing)} prompts, skipping phase0 generation")
            return {}

        print("\n🔄 Phase 0: Generating initial variants via LLM...")

        try:
            improver = PromptImprover(
                api_base=llm_config.get('meta_llm_api_base', 'http://localhost:8000/v1'),
                model_name=llm_config.get('meta_llm_model', 'Qwen/Qwen2.5-VL-72B-Instruct'),
                api_key=llm_config.get('meta_llm_api_key', 'dummy'),
                temperature=0.7,
                use_vision=False,
            )

            meta_instructions_dir = self.instructions_dir.parent / "meta_instructions"
            phase0_file = meta_instructions_dir / "phase0_init_variants.txt"

            if not phase0_file.exists():
                print(f"⚠️ phase0 meta instruction not found: {phase0_file}")
                return {}

            phase0_prompt = read_text_file(phase0_file)
            generation_prompt = phase0_prompt.replace("[VANILLA_INSTRUCTION]", base_prompt)

            import requests
            response = requests.post(
                f"{improver.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {improver.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": improver.model_name,
                    "messages": [{"role": "user", "content": [{"type": "text", "text": generation_prompt}]}],
                    "temperature": improver.temperature,
                    "max_tokens": 2048,
                    "top_p": 0.9,
                },
                timeout=180,
            )

            if response.status_code != 200:
                print(f"❌ LLM API error ({response.status_code}): {response.text[:200]}")
                return {}

            result = response.json()
            if 'choices' not in result or not result['choices']:
                print(f"❌ Invalid API response: {result}")
                return {}

            response_text = result['choices'][0]['message']['content']
            variants = self._parse_variants_from_response(response_text)

            variants_dir.mkdir(parents=True, exist_ok=True)
            for name, content in variants.items():
                try:
                    write_text_file(content, variants_dir / f"{name}.txt")
                    print(f"✓ Generated variant '{name}': {len(content)} characters")
                except Exception as e:
                    print(f"⚠️ Failed to save variant '{name}': {e}")

            return variants

        except Exception as e:
            print(f"❌ Phase 0 generation failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _parse_variants_from_response(self, response_text: str) -> Dict[str, str]:
        variants = {}
        pattern = r'Variant\s+(\d+):\s*\[STRATEGY:\s*([^\]]+)\](.*?)(?=Variant\s+\d+:|$)'
        matches = re.finditer(pattern, response_text, re.DOTALL | re.IGNORECASE)

        for match in matches:
            variant_num = match.group(1)
            strategy = match.group(2).strip()
            content = match.group(3).strip()
            name = strategy.lower().replace(" ", "_") if strategy else f"variant_{variant_num}"
            name = name.split('_strategy')[0]
            if content:
                variants[name] = content

        if not variants:
            print("⚠️ Standard pattern failed, trying fallback parsing...")
            lines = response_text.split('\n')
            current_variant = None
            current_content = []
            for line in lines:
                if line.startswith('Variant') and ':' in line:
                    if current_variant and current_content:
                        variants[current_variant] = '\n'.join(current_content).strip()
                    current_variant = f"variant_{len(variants) + 1}"
                    current_content = []
                elif current_variant and line.strip():
                    current_content.append(line)
            if current_variant and current_content:
                variants[current_variant] = '\n'.join(current_content).strip()

        return variants

    def save_prompt(self, name: str, prompt: str, metadata: Dict = None) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.output_dir / f"prompt_{name}_{timestamp}.txt"
        write_text_file(prompt, filepath)
        if metadata:
            save_json(metadata, self.output_dir / f"meta_{name}_{timestamp}.json")
        self.history.append({
            'name': name,
            'timestamp': timestamp,
            'filepath': str(filepath),
            'metadata': metadata,
        })
        print(f"✓ Saved prompt: {filepath}")
        return filepath

    def update_score(self, prompt_name: str, score: float) -> None:
        self.scores[prompt_name] = score
