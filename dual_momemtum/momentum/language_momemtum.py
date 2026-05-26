from typing import List, Optional

from modules.llm_optimizer import LLMOptimizer


class LanguageMomemtumOptimizer(LLMOptimizer):
    """Language momentum optimizer.

    Drives the language_momemtum phase: given a population of scored prompts,
    performs contrastive Top-K vs Bottom-K distribution analysis and generates
    an improved prompt via the meta-LLM.
    """

    def analyze_distribution(self, base_text: str, base_score: float,
                              better_prompts: List[str], worse_prompts: List[str],
                              better_scores: List[float], worse_scores: List[float],
                              iteration: int) -> Optional[str]:
        """Analyze prompt distribution and generate an improved prompt.

        If all 'worse' prompts still outperform the base, treat the whole pool
        as successful candidates and synthesise further improvements.
        Otherwise, do standard high-vs-low contrastive analysis.
        """
        analysis_prompt = self._build_analysis_prompt(
            base_text, base_score,
            better_prompts, worse_prompts,
            better_scores, worse_scores,
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": analysis_prompt}]}]

        print(f"🔄 LanguageMomemtum: analyzing distribution (iteration {iteration})...", flush=True)

        response_text = self._call_api(messages, iteration)
        if response_text is None:
            return None

        print(f"response: {response_text}")

        improved = self._extract_improved_prompt(response_text)
        if improved:
            print("✅ LanguageMomemtum: improved prompt generated")
        else:
            print(f"⚠️ LanguageMomemtum: could not extract prompt from response")
        return improved

    def _build_analysis_prompt(self, base_text: str, base_score: float,
                                better_prompts: List[str], worse_prompts: List[str],
                                better_scores: List[float], worse_scores: List[float]) -> str:
        worse_all_beat_base = all(s >= base_score for s in worse_scores)

        if worse_all_beat_base:
            return f"""You are analyzing prompts for medical image analysis tasks. All candidate prompts perform better than the baseline, so analyze what makes them successful and generate an even better version.

BASE PROMPT (Baseline Reference):
Score: {base_score:.4f}
---
{base_text}
---

BEST PERFORMING PROMPTS (Currently Highest Quality):
{self._format_comparison_prompts(better_prompts, better_scores, base_score, "best")}

RELATIVELY WEAKER PROMPTS (Still Better than Base, but Less Effective):
{self._format_comparison_prompts(worse_prompts, worse_scores, base_score, "weaker")}

YOUR TASK:
1. Analyze what makes the best prompts work so well
2. Identify the key success factors in both prompt sets
3. Understand what distinguishes the best from the weaker ones
4. Generate ONE improved prompt that combines the best elements and addresses any remaining weaknesses

Output in this format:
<ANALYSIS>
[Analysis of what makes these prompts successful and how to improve further]
</ANALYSIS>

<IMPROVED_PROMPT>
[Your improved prompt incorporating best elements from both sets]
</IMPROVED_PROMPT>
"""
        else:
            return f"""You are analyzing prompts for medical image analysis tasks. Your task is to understand what makes some prompts work better and others worse, then generate an improved version.

BASE PROMPT (Performance Reference):
Score: {base_score:.4f}
---
{base_text}
---

HIGH PERFORMANCE PROMPTS (What Works Well):
{self._format_comparison_prompts(better_prompts, better_scores, base_score, "better")}

LOW PERFORMANCE PROMPTS (What Doesn't Work):
{self._format_comparison_prompts(worse_prompts, worse_scores, base_score, "worse")}

YOUR TASK:
1. Analyze the key differences between high and low performing prompts
2. Identify what specific elements in high-performance prompts contribute to success
3. Understand what aspects in low-performance prompts hurt performance
4. Based on these insights, generate ONE improved prompt that incorporates successful elements while avoiding failures.
5. The output prompt should not be too long.

Output in this format:
<IMPROVED_PROMPT>
[Your improved prompt incorporating successful elements and avoiding pitfalls]
</IMPROVED_PROMPT>
"""
