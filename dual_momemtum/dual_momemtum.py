import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from modules import PromptManager, DataManager, TargetModelRunner, Evaluator, PromptPopulation
from modules.logger import get_logger
from momentum import LanguageMomemtumOptimizer, VisualMomemtumOptimizer, ConvergenceHandler

logger = get_logger(__name__)


class DualMomemtumSearcher:
    """Dual Momemtum prompt search framework.

    Orchestrates language_momemtum (LLM distribution analysis) and
    visual_momemtum (visual error feedback), switching modes via convergence_handler.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 70)
        print("Initializing Dual Momemtum Prompt Search")
        print("=" * 70)

        print("\n[1/6] Loading prompts...")
        self.prompt_manager = PromptManager(config['instructions_dir'], str(self.output_dir))
        base_prompt = self.prompt_manager.load_base_prompt()

        # Phase 0: auto-generate initial variants if none exist
        llm_config = {
            'meta_llm_api_base': config.get('meta_llm_api_base'),
            'meta_llm_model': config.get('meta_llm_model'),
            'meta_llm_api_key': config.get('meta_llm_api_key'),
        }
        self.prompt_manager.generate_initial_variants_if_needed(base_prompt, llm_config)
        variants = self.prompt_manager.load_variant_prompts()

        print("\n[2/6] Loading data...")
        self.data_manager = DataManager(
            config['train_data_path'], config['dev_data_path'], config['test_data_path']
        )

        print("\n[3/6] Initializing target model...")
        self.target_runner = TargetModelRunner(
            config['target_model_path'],
            gpu_memory_utilization=config.get('gpu_memory_utilization', 0.9),
            batch_size=config.get('batch_size', 64),
            tensor_parallel_size=config.get('tensor_parallel_size'),
            image_path_replacement=config.get('image_path_replacement'),
        )
        print(f"seed: {config.get('seed', 42)}")
        self.target_runner.init_model(
            temperature=config.get('temperature', 0.0),
            top_p=config.get('top_p', 1.0),
            max_gen_tokens=config.get('max_gen_tokens', 512),
            seed=config.get('seed', 42),
        )

        print("\n[4/6] Initializing evaluator...")
        self.evaluator = Evaluator()

        print("\n[5/6] Initializing optimizers...")
        self.language_momemtum = LanguageMomemtumOptimizer(
            api_base=config.get('meta_llm_api_base', 'http://localhost:8000/v1'),
            model_name=config.get('meta_llm_model', 'Qwen/Qwen2.5-VL-72B-Instruct'),
            api_key=config.get('meta_llm_api_key', 'dummy'),
            temperature=config.get('meta_llm_temperature', 0.7),
            use_vision=config.get('meta_llm_use_vision', True),
            output_dir=str(self.output_dir),
        )
        self.visual_momemtum = VisualMomemtumOptimizer(
            api_base=config.get('meta_llm_api_base', 'http://localhost:8000/v1'),
            model_name=config.get('meta_llm_model', 'Qwen/Qwen2.5-VL-72B-Instruct'),
            api_key=config.get('meta_llm_api_key', 'dummy'),
            temperature=config.get('meta_llm_temperature', 0.7),
            output_dir=str(self.output_dir / 'visualizations'),
        )
        self.convergence_handler = ConvergenceHandler(
            convergence_threshold=config.get('convergence_threshold', 0.0001),
            convergence_window=config.get('convergence_window', 3),
            enable_mode_switch=config.get('enable_visual_feedback', True),
            allow_switch_back=config.get('allow_switch_back_to_tpe', False),
            visual_plateau_window=config.get('visual_plateau_window', 5),
        )

        print("\n[6/6] Initializing prompt population...")
        self.population = PromptPopulation()
        self.population.add_prompt('base', base_prompt, source='base.txt')
        for name, prompt in variants.items():
            self.population.add_prompt(name, prompt, source='variant')

        self.generated_prompts_history = {}
        print("\n✅ Ready\n")

    def run_dual_momemtum(self, num_iterations: int = 5) -> Tuple[str, float]:
        print("\n" + "=" * 70)
        print("Starting Dual Momemtum Prompt Search")
        print("=" * 70)

        best_prompt = self.population.population['base']['text']
        best_score = None
        max_patience = self.config.get('max_patience', 10)
        consecutive_no_improvement = 0
        convergence_history = []

        for iteration in range(num_iterations):
            print(f"\n{'='*70}")
            print(f"Iteration {iteration + 1}/{num_iterations}")
            print(f"{'='*70}")

            self._current_iteration = iteration

            try:
                if iteration == 0:
                    print("\n[Step 0] Evaluating initial prompt population...")
                    self._evaluate_all_prompts()
                    top = self.population.get_top_prompts(k=1)
                    if top:
                        best_score = top[0][1].get('score', 0.0)
                        best_prompt = top[0][1].get('text', best_prompt)
                        print(f"   Initial baseline score: {best_score:.4f}")

                print("\n[Step 1] Population stats and comparison selection...")
                base_data = self.population.population.get('base', {})
                base_score = base_data.get('score', 0.0)
                base_text = base_data.get('text', '')
                print(f"   Base prompt score: {base_score:.4f}")

                stats = self.population.get_distribution_stats()
                print(f"   Population: count={stats.get('count', 0)}, "
                      f"mean={stats.get('mean', 0):.4f}, "
                      f"max={stats.get('max', 0):.4f}, "
                      f"min={stats.get('min', 0):.4f}")

                all_non_base = [
                    (name, data) for name, data in self.population.population.items()
                    if name != 'base'
                ]
                all_non_base_sorted = sorted(
                    all_non_base, key=lambda x: x[1].get('score', 0), reverse=True
                )

                if not all_non_base_sorted:
                    print("No other prompts in population, skipping iteration")
                    continue

                dynamic_k_max = self.config.get('dynamic_k_max', 5)
                k = min(
                    (iteration // 2) + 1,
                    len(all_non_base_sorted) // 2,
                    dynamic_k_max,
                )

                print(f"   Dynamic k: iter {iteration+1} → Top-{k} vs Bottom-{k} (max_k={dynamic_k_max})")
                better_prompts = all_non_base_sorted[:k]
                worse_prompts = list(reversed(all_non_base_sorted))[:k]

                better_names = [p[0] for p in better_prompts]
                better_data = [p[1] for p in better_prompts]
                better_scores = [d.get('score', 0) for d in better_data]
                worse_names = [p[0] for p in worse_prompts]
                worse_data = [p[1] for p in worse_prompts]
                worse_scores = [d.get('score', 0) for d in worse_data]

                print(f"\n   Top-{k}:")
                for i, (name, score) in enumerate(zip(better_names, better_scores)):
                    diff = score - base_score
                    print(f"     {i+1}. {name}: {score:.4f} ({'+' if diff>=0 else ''}{diff:.4f})")
                print(f"\n   Bottom-{k}:")
                for i, (name, score) in enumerate(zip(worse_names, worse_scores)):
                    diff = score - base_score
                    print(f"     {i+1}. {name}: {score:.4f} ({'+' if diff>=0 else ''}{diff:.4f})")

                print("\n[Step 2] Generating new prompt based on current mode...")

                current_mode = self.convergence_handler.get_current_mode()
                ablation = self.config.get('ablation_mode')

                if ablation == 'language_only':
                    current_mode = 'language_momemtum'
                    print("   [Ablation] Locked to LanguageMomemtum")
                elif ablation == 'visual_only':
                    current_mode = 'visual_momemtum'
                    print("   [Ablation] Locked to VisualMomemtum")

                if current_mode == 'language_momemtum':
                    print("   Mode: LanguageMomemtum")
                    new_prompt = self.language_momemtum.analyze_distribution(
                        base_text=base_text,
                        base_score=base_score,
                        better_prompts=[d['text'] for d in better_data],
                        worse_prompts=[d['text'] for d in worse_data],
                        better_scores=better_scores,
                        worse_scores=worse_scores,
                        iteration=iteration + 1,
                    )

                    if new_prompt is None:
                        print("⚠️ Meta-LLM returned no valid prompt, skipping")
                        consecutive_no_improvement += 1
                        if consecutive_no_improvement >= max_patience:
                            print(f"Meta-LLM failed {consecutive_no_improvement} times, stopping")
                            break
                        continue

                    print("\n[Step 3] Evaluating new prompt...")
                    new_score = self._evaluate_prompt(new_prompt)

                    score_key = round(new_score, 6)
                    self.generated_prompts_history.setdefault(score_key, []).append(new_prompt)

                    self.population.add_prompt(
                        f'language_momemtum_gen_{iteration}', new_prompt,
                        score=new_score, source='language_momemtum_generated',
                    )

                else:
                    print("   Mode: VisualMomemtum")
                    vf_result = self._run_visual_momemtum_loop(
                        best_prompt=best_prompt, best_score=best_score,
                        base_text=base_text, iteration=iteration,
                        max_patience=self.config.get('visual_sub_loop_patience', 3),
                    )
                    if vf_result is None:
                        print("⚠️ VisualMomemtum sub-loop failed")
                        consecutive_no_improvement += 1
                        continue

                    new_prompt = vf_result['best_prompt']
                    new_score = vf_result['best_score']

                    for gen_prompt, gen_score, gen_name in vf_result['generated_prompts']:
                        self.population.add_prompt(
                            gen_name, gen_prompt, score=gen_score, source='visual_momemtum'
                        )

                print(f"\n   Best score this round: {new_score:.4f}")
                if current_mode == 'language_momemtum':
                    print(f"   Generated: language_momemtum_gen_{iteration}")
                else:
                    print(f"   VisualMomemtum generated {len(vf_result['generated_prompts'])} prompts:")
                    for i, (_, score, name) in enumerate(vf_result['generated_prompts'], 1):
                        print(f"     {i}. {name}: score={score:.4f}")

                if new_score > best_score:
                    improvement = new_score - best_score
                    best_score = new_score
                    best_prompt = new_prompt
                    consecutive_no_improvement = 0
                    print(f"   ✨ New best! +{improvement:.4f} → {best_score:.4f}")
                else:
                    consecutive_no_improvement += 1
                    print(f"   No improvement ({new_score:.4f} vs best {best_score:.4f}) "
                          f"[consecutive: {consecutive_no_improvement}/{max_patience}]")
                    if consecutive_no_improvement >= max_patience:
                        print(f"No improvement for {consecutive_no_improvement} rounds, stopping")
                        break

                self.population.generation += 1

                print(f"\n[Step 4] Updated population stats:")
                updated_stats = self.population.get_distribution_stats()
                updated_top = self.population.get_top_prompts(k=3)
                updated_bottom = self.population.get_bottom_prompts(k=3)
                print(f"   count={updated_stats.get('count', 0)}, "
                      f"mean={updated_stats.get('mean', 0):.4f}, "
                      f"max={updated_stats.get('max', 0):.4f}, "
                      f"std={updated_stats.get('std', 0):.4f}, "
                      f"top3_std={updated_stats.get('top_3_std', 0):.6f}")
                print("   Top-3:")
                for i, (name, data) in enumerate(updated_top):
                    print(f"     {i+1}. {name}: {data.get('score', 0):.4f}")
                print("   Bottom-3:")
                for i, (name, data) in enumerate(updated_bottom):
                    print(f"     {i+1}. {name}: {data.get('score', 0):.4f}")

                has_improvement = new_score > (getattr(self, 'last_best_score', best_score))
                self.last_best_score = best_score

                convergence_result = self.convergence_handler.check_and_update(
                    updated_stats.get('top_3_std', 0), iteration, has_improvement=has_improvement
                )

                if ablation == 'language_only':
                    convergence_result['mode'] = 'language_momemtum'
                    convergence_result['should_switch'] = False
                    convergence_result['reason'] = "[Ablation] Forced LanguageMomemtum"
                elif ablation == 'visual_only':
                    convergence_result['mode'] = 'visual_momemtum'
                    convergence_result['should_switch'] = False
                    convergence_result['reason'] = "[Ablation] Forced VisualMomemtum"

                convergence_history.append(convergence_result)
                self._save_convergence_record(convergence_result)

                print(f"\n[Step 5] Mode management:")
                print(f"   Current mode: {convergence_result['mode'].upper()}")
                print(f"   Top-3 std: {convergence_result['top_3_std']:.6f}")
                print(f"   {convergence_result['reason']}")
                print(f"   {convergence_result['recommendation']}")
                if convergence_result['should_switch']:
                    print(f"   Switch triggered: {convergence_result.get('switch_reason', '')}")

            except Exception as e:
                print(f"\nIteration {iteration + 1} error: {e}")
                traceback.print_exc()
                continue

        print("\n" + "=" * 70)
        print("✅ Dual Momemtum search complete!")
        print("=" * 70)
        print(f"\nBest score: {best_score:.4f}")
        print(f"Best prompt preview:\n{best_prompt[:200]}...")

        self._save_population()
        self._save_convergence_report(convergence_history)

        return best_prompt, best_score

    # -------------------------------------------------------------------------
    # Evaluation helpers
    # -------------------------------------------------------------------------

    def _evaluate_all_prompts(self) -> None:
        for name, data in self.population.population.items():
            if data.get('score') is None:
                print(f"   Evaluating '{name}'...", end="", flush=True)
                try:
                    score = self._evaluate_prompt(data['text'])
                    self.population.population[name]['score'] = score
                    print(f" score: {score:.4f}")
                except Exception as e:
                    print(f" error: {e}")
                    self.population.population[name]['score'] = 0.0

    def _evaluate_prompt(self, prompt: str) -> float:
        all_scores = []
        all_errors = []
        total_samples = 0

        prompt_hash = str(hash(prompt))[-8:]
        prompt_name = f"prompt_{prompt_hash}"

        logger.info(f"\n{'='*60}\nEvaluating prompt\n{prompt[:200]}...\n{'='*60}")

        try:
            for mini_batch, batch_idx, total_batches in self.data_manager.get_dev_iterator(
                self.config.get('batch_size', 32)
            ):
                infer_results = self.target_runner.infer_batch(mini_batch, prompt)

                if not infer_results:
                    print(f"     Batch {batch_idx + 1}/{total_batches}: 0 valid samples",
                          end="\r", flush=True)
                    continue

                self._save_inference_results(
                    infer_results, getattr(self, '_current_iteration', 0), prompt_name
                )

                batch_scores, batch_errors = self.evaluator.evaluate_batch(
                    infer_results,
                    coord_mode=self.config.get('coord_mode', 'pixel'),
                    path_from=self.config.get('path_from'),
                    path_to=self.config.get('path_to'),
                )
                all_scores.extend(batch_scores)
                all_errors.extend(batch_errors)
                total_samples += len(infer_results)

                overall = sum(all_scores) / len(all_scores)
                logger.info(f"Batch {batch_idx+1}/{total_batches}: {len(infer_results)} samples, "
                            f"overall mAP: {overall:.4f}")
                print(f"     Progress: {batch_idx+1}/{total_batches} | "
                      f"samples: {total_samples} | "
                      f"avg mAP: {overall:.4f}",
                      end="\r", flush=True)

            avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
            logger.info(f"Evaluation done: {total_samples} samples, mAP@0.5={avg_score:.4f}")

            self.last_iteration_errors = all_errors
            self.last_iteration_error_summary = (
                self.evaluator.summarize_errors(all_errors) if all_errors else {}
            )

            print(f"     Done! samples: {total_samples} | mAP@0.5: {avg_score:.4f}          ")
            return avg_score

        except Exception as e:
            logger.error(f"Evaluation failed: {e}", exc_info=True)
            print(f"\n     Evaluation failed: {e}")
            return 0.0

    def _save_inference_results(self, results: List[Dict], iteration: int, prompt_name: str) -> None:
        if not hasattr(self, '_inference_results_dir'):
            self._inference_results_dir = self.output_dir / "inference_results"
            self._inference_results_dir.mkdir(parents=True, exist_ok=True)

        safe_name = prompt_name.replace('/', '_').replace(' ', '_')[:30]
        results_file = self._inference_results_dir / f"inference_iter{iteration:02d}_{safe_name}.jsonl"

        try:
            with open(results_file, 'a', encoding='utf-8') as f:
                for result in results:
                    json.dump({
                        'iteration': iteration,
                        'prompt_name': prompt_name,
                        'timestamp': datetime.now().isoformat(),
                        'sample_idx': result['sample_idx'],
                        'image_path': result['image_path'],
                        'answer': result['answer'],
                        'parsed_boxes': result['parsed_boxes'],
                    }, f, ensure_ascii=False)
                    f.write('\n')
        except Exception as e:
            print(f"⚠️ Failed to save inference results: {e}")

    # -------------------------------------------------------------------------
    # VisualMomemtum sub-loop
    # -------------------------------------------------------------------------

    def _run_visual_momemtum_loop(self, best_prompt: str, best_score: float,
                                   base_text: str, iteration: int,
                                   max_patience: int = 5) -> Optional[Dict[str, Any]]:
        print(f"\n   ===== VisualMomemtum sub-loop start =====")

        if not hasattr(self, 'last_iteration_errors') or not self.last_iteration_errors:
            print("   No error samples available")
            return None

        current_best_prompt = best_prompt
        current_best_score = best_score
        generated_prompts = []
        vf_no_improvement = 0

        for vf_iter in range(max_patience):
            print(f"\n   [VF_iter_{vf_iter + 1}] current best={current_best_score:.4f}", flush=True)

            new_prompt = self.visual_momemtum.improve_with_visual_feedback(
                current_prompt=current_best_prompt,
                errors=self.last_iteration_errors,
                error_summary=getattr(self, 'last_iteration_error_summary', {}),
                iteration=f"{iteration}_vf{vf_iter + 1}",
                base_prompt=base_text,
            )

            if new_prompt is None:
                vf_no_improvement += 1
                if vf_no_improvement >= max_patience:
                    print(f"     {vf_no_improvement} consecutive failures, stopping")
                    break
                continue

            new_score = self._evaluate_prompt(new_prompt)
            vf_prompt_name = f'visual_momemtum_gen_{iteration}_vf{vf_iter + 1}'
            generated_prompts.append((new_prompt, new_score, vf_prompt_name))
            self.generated_prompts_history.setdefault(round(new_score, 6), []).append(new_prompt)

            print(f"     New score: {new_score:.4f} | Best: {current_best_score:.4f}", flush=True)

            if new_score > current_best_score:
                improvement = new_score - current_best_score
                current_best_prompt = new_prompt
                current_best_score = new_score
                vf_no_improvement = 0
                print(f"     ✨ +{improvement:.4f} — returning to LanguageMomemtum")
                break
            else:
                vf_no_improvement += 1
                if vf_no_improvement >= max_patience:
                    print(f"     {vf_no_improvement} rounds without improvement, stopping")
                    break

        print(f"   ===== VisualMomemtum sub-loop end: "
              f"{len(generated_prompts)} generated, best={current_best_score:.4f} =====")

        return {
            'best_prompt': current_best_prompt,
            'best_score': current_best_score,
            'generated_prompts': generated_prompts,
        }

    # -------------------------------------------------------------------------
    # Persistence helpers
    # -------------------------------------------------------------------------

    def _save_population(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        population_file = self.output_dir / f"population_{timestamp}.json"
        population_data = {
            'timestamp': timestamp,
            'generation': self.population.generation,
            'population': {
                name: {
                    'text': data.get('text', ''),
                    'score': data.get('score'),
                    'source': data.get('source'),
                    'generation': data.get('generation'),
                }
                for name, data in self.population.population.items()
            },
        }
        with open(population_file, 'w', encoding='utf-8') as f:
            json.dump(population_data, f, indent=2, ensure_ascii=False)
        print(f"\nPopulation saved: {population_file}")

    def _save_convergence_record(self, record: Dict) -> None:
        if not hasattr(self, '_convergence_log_file'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._convergence_log_file = self.output_dir / f"convergence_log_{timestamp}.jsonl"

        top_3 = self.population.get_top_prompts(k=3)
        bottom_3 = self.population.get_bottom_prompts(k=3)

        complete_record = {
            'iteration': record.get('iteration', -1),
            'mode': record.get('mode', 'unknown'),
            'is_converged': record.get('is_converged', False),
            'top_3_std': record.get('top_3_std', 0),
            'top_3_scores': [data.get('score', 0) for _, data in top_3],
            'threshold': self.convergence_handler.convergence_threshold,
            'reason': record.get('reason', ''),
            'recommendation': record.get('recommendation', ''),
            'timestamp': datetime.now().isoformat(),
            'prompts_snapshot': {
                'top_3': [{'name': n, 'text': d.get('text', ''),
                           'score': d.get('score'), 'source': d.get('source')}
                          for n, d in top_3],
                'bottom_3': [{'name': n, 'text': d.get('text', ''),
                              'score': d.get('score'), 'source': d.get('source')}
                             for n, d in bottom_3],
            },
        }

        try:
            with open(self._convergence_log_file, 'a', encoding='utf-8') as f:
                json.dump(complete_record, f, ensure_ascii=False)
                f.write('\n')
        except Exception as e:
            print(f"⚠️ Failed to save convergence record: {e}")

    def _save_convergence_report(self, convergence_history: List[Dict]) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"convergence_report_{timestamp}.json"

        final_history = convergence_history
        if hasattr(self, '_convergence_log_file') and self._convergence_log_file.exists():
            final_history = []
            try:
                with open(self._convergence_log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            final_history.append(json.loads(line))
            except Exception as e:
                print(f"⚠️ Failed to read convergence log: {e}")
                final_history = convergence_history

        report = {
            'timestamp': timestamp,
            'total_iterations': len(final_history),
            'convergence_detected': any(c.get('is_converged', False) for c in final_history),
            'first_convergence_at': next(
                (c['iteration'] for c in final_history if c.get('is_converged')), None
            ),
            'convergence_history': final_history,
            'final_summary': {
                'reason': final_history[-1]['reason'] if final_history else 'N/A',
                'recommendation': final_history[-1]['recommendation'] if final_history else 'N/A',
            },
        }

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nConvergence report saved: {report_file}")

        if report['convergence_detected']:
            first_conv = report['first_convergence_at']
            print(f"Convergence first detected at iteration {first_conv}")
            print(f"Final recommendation: {report['final_summary']['recommendation']}")
        else:
            print("No convergence detected, diversity maintained")
