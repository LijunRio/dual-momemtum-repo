#!/usr/bin/env python3

from typing import Dict, Tuple


class ConvergenceHandler:
    """
    Convergence detection and mode switching between language_momemtum and visual_momemtum.

    Logic:
    - Iter 0+: language_momemtum mode
    - When Top-3 std < threshold for N consecutive rounds → switch to visual_momemtum
    - Optionally switch back to language_momemtum if visual_momemtum stagnates
    """

    def __init__(self,
                 convergence_threshold: float = 0.0001,
                 convergence_window: int = 3,
                 enable_mode_switch: bool = True,
                 allow_switch_back: bool = False,
                 visual_plateau_window: int = 5):
        self.convergence_threshold = convergence_threshold
        self.convergence_window = convergence_window
        self.enable_mode_switch = enable_mode_switch
        self.allow_switch_back = allow_switch_back
        self.visual_plateau_window = visual_plateau_window

        self.mode = 'language_momemtum'
        self.convergence_count = 0
        self.visual_no_improvement_count = 0
        self.last_convergence_status = None
        self.switched_to_visual_at = None

    def check_and_update(self, top_3_std: float, iteration: int,
                         has_improvement: bool = True) -> Dict:
        old_mode = self.mode
        should_switch = False
        switch_reason = ""

        if old_mode == 'language_momemtum':
            is_converged = top_3_std < self.convergence_threshold

            if is_converged:
                self.convergence_count += 1
            else:
                self.convergence_count = 0

            should_switch = (
                self.enable_mode_switch and
                self.convergence_count >= self.convergence_window
            )

            if should_switch:
                self.mode = 'visual_momemtum'
                self.switched_to_visual_at = iteration
                self.visual_no_improvement_count = 0
                self.convergence_count = 0
                switch_reason = f"LanguageMomemtum converged ({self.convergence_window} rounds) → VisualMomemtum"

        else:  # visual_momemtum
            if not has_improvement:
                self.visual_no_improvement_count += 1
            else:
                self.visual_no_improvement_count = 0

            should_switch_back = (
                self.allow_switch_back and
                self.visual_no_improvement_count >= self.visual_plateau_window
            )

            if should_switch_back:
                self.mode = 'language_momemtum'
                self.convergence_count = 0
                self.visual_no_improvement_count = 0
                should_switch = True
                switch_reason = f"VisualMomemtum stagnated ({self.visual_plateau_window} rounds) → LanguageMomemtum"

            is_converged = False

        reason, recommendation = self._get_diagnosis_v2(
            old_mode,
            is_converged if old_mode == 'language_momemtum' else None,
            top_3_std,
            self.convergence_count,
            self.visual_no_improvement_count,
            should_switch,
            switch_reason
        )

        self.last_convergence_status = {
            'iteration': iteration,
            'is_converged': is_converged if old_mode == 'language_momemtum' else False,
            'top_3_std': top_3_std,
            'mode': self.mode,
            'should_switch': should_switch,
            'switch_reason': switch_reason,
            'reason': reason,
            'recommendation': recommendation
        }

        return self.last_convergence_status

    def _get_diagnosis_v2(self, mode: str, is_converged: bool, top_3_std: float,
                          convergence_count: int, visual_no_improvement_count: int,
                          should_switch: bool, switch_reason: str) -> Tuple[str, str]:
        if mode == 'language_momemtum':
            if should_switch:
                reason = switch_reason
                recommendation = "Switch to VisualMomemtum — optimizing via visual error analysis"
            elif is_converged:
                reason = f"Convergence signal #{convergence_count} (Top-3 std={top_3_std:.6f})"
                recommendation = (f"Monitoring convergence — "
                                  f"{self.convergence_window - convergence_count} more rounds until switch")
            else:
                reason = f"No convergence (Top-3 std={top_3_std:.6f})"
                recommendation = "Continue LanguageMomemtum distribution analysis"
        else:  # visual_momemtum
            if should_switch:
                reason = switch_reason
                recommendation = "Switch back to LanguageMomemtum — VisualMomemtum stagnated"
            elif visual_no_improvement_count > 0:
                reason = f"VisualMomemtum no improvement ({visual_no_improvement_count}/{self.visual_plateau_window})"
                if self.allow_switch_back:
                    remaining = self.visual_plateau_window - visual_no_improvement_count
                    recommendation = f"Continue VisualMomemtum — switch back in {remaining} more rounds if no improvement"
                else:
                    recommendation = "Continue VisualMomemtum optimization"
            else:
                reason = "VisualMomemtum running"
                recommendation = "Continue visual feedback optimization"

        return reason, recommendation

    def get_current_mode(self) -> str:
        return self.mode

    def reset(self):
        self.mode = 'language_momemtum'
        self.convergence_count = 0
        self.last_convergence_status = None

    def get_status_summary(self) -> Dict:
        return {
            'mode': self.mode,
            'convergence_count': self.convergence_count,
            'convergence_window': self.convergence_window,
            'convergence_threshold': self.convergence_threshold,
            'last_status': self.last_convergence_status
        }
