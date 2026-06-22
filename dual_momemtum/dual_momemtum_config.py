#!/usr/bin/env python3

from pathlib import Path
import copy

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = Path(__file__).parent / "dual_momemtum_results"

DUAL_MOMEMTUM_CONFIG = {
    # Dataset (relative to repo root data/)
    'train_data_path': str(DATA_ROOT / 'dev_nova_100.json'),
    'dev_data_path': str(DATA_ROOT / 'dev_nova_100.json'),
    'test_data_path': str(DATA_ROOT / 'dev_nova_100.json'),

    # Target model
    'target_model_path': '/home/june/cache/huggingface_checkpoints/Qwen2.5-VL-3B-Instruct',

    # Inference
    'batch_size': 32,
    'temperature': 0.0,
    'top_p': 1.0,
    'max_gen_tokens': 512,
    'max_tokens': 4096,
    'seed': 0,

    # GPU
    'gpu_memory_utilization': 0.95,
    'tensor_parallel_size': None,

    # Image path replacement
    'image_path_replacement': {
        'from': '/home/june/datasets/nova',
        'to': '/home/june/datasets/nova'
    },

    # Evaluation
    'coord_mode': 'pixel',
    'path_from': '/home/june/datasets/nova',
    'path_to': '/home/june/datasets/nova',

    # Meta-LLM (used by both language_momemtum and visual_momemtum)
    'meta_llm_api_base': "https://openrouter.ai/api/v1",
    'meta_llm_model': 'openai/gpt-4o-2024-11-20', # google/gemini-3-flash-preview
    'meta_llm_api_key': 'You-API-Key',
    'meta_llm_temperature': 0.7,
    'meta_llm_use_vision': True,  # visual_momemtum requires multimodal capability


    # Search parameters
    'num_iterations': 5, #standard 20  test 5
    'top_k': 3,
    'bottom_k': 3,
    'max_patience': 20,
    'dynamic_k_max': 10,
    'instructions_dir': str(Path(__file__).parent / 'instructions'),
    'output_dir': str(OUTPUT_ROOT),

    # Convergence and mode switching
    'convergence_threshold': 0.005,
    'convergence_window': 3,
    'enable_visual_feedback': True,
    'allow_switch_back_to_tpe': True,
    'visual_plateau_window': 5,
    'visual_sub_loop_patience': 5, # test 3
    'ablation_mode': None,  # None | 'language_only' | 'visual_only'
}


def get_config() -> dict:
    return copy.deepcopy(DUAL_MOMEMTUM_CONFIG)


def validate_config(config: dict) -> bool:
    required_keys = [
        'target_model_path',
        'dev_data_path',
        'meta_llm_api_base',
        'meta_llm_model',
        'meta_llm_api_key',
        'instructions_dir',
        'output_dir'
    ]

    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        print(f"Missing required config keys: {missing_keys}")
        return False

    dev_path = Path(config['dev_data_path'])
    if not dev_path.exists():
        print(f"Dev data path not found: {dev_path}")
        return False

    instructions_dir = Path(config['instructions_dir'])
    if not (instructions_dir / 'base.txt').exists():
        print(f"base.txt not found: {instructions_dir / 'base.txt'}")
        return False

    return True
