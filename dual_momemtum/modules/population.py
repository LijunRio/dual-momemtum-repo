from typing import Dict, List, Tuple


class PromptPopulation:

    def __init__(self):
        self.population = {}
        self.generation = 0

    def add_prompt(self, name: str, text: str, score: float = None, source: str = None):
        self.population[name] = {
            'text': text,
            'score': score,
            'source': source,
            'generation': self.generation,
        }

    def get_top_prompts(self, k: int = 5) -> List[Tuple[str, Dict]]:
        return sorted(
            self.population.items(),
            key=lambda x: x[1].get('score', -float('inf')),
            reverse=True
        )[:k]

    def get_bottom_prompts(self, k: int = 5) -> List[Tuple[str, Dict]]:
        return sorted(
            self.population.items(),
            key=lambda x: x[1].get('score', float('inf'))
        )[:k]

    def get_distribution_stats(self) -> Dict:
        scores = [p.get('score', 0) for p in self.population.values() if p.get('score') is not None]
        if not scores:
            return {'count': 0}

        mean = sum(scores) / len(scores)
        std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
        sorted_scores = sorted(scores, reverse=True)
        top_3 = sorted_scores[:3]
        top_5 = sorted_scores[:5]

        def _std(vals):
            if not vals:
                return 0
            m = sum(vals) / len(vals)
            return (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5

        return {
            'count': len(scores),
            'mean': mean,
            'min': min(scores),
            'max': max(scores),
            'std': std,
            'top_3_std': _std(top_3),
            'top_5_std': _std(top_5),
            'top_3_scores': top_3,
            'top_5_scores': top_5,
        }
