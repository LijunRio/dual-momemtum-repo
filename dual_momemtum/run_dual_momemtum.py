#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from modules.logger import setup_logging, get_logger
from dual_momemtum import DualMomemtumSearcher
from dual_momemtum_config import get_config, validate_config


def main():
    parser = argparse.ArgumentParser(
        description='Dual Momemtum prompt optimization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_dual_momemtum.py
  python run_dual_momemtum.py -n 10
  python run_dual_momemtum.py --api-key YOUR_KEY
        """
    )
    parser.add_argument('-n', '--iterations', type=int, default=None)
    parser.add_argument('--api-key', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--model-path', type=str, default=None)
    parser.add_argument('--train-data', type=str, default=None)
    parser.add_argument('--dev-data', type=str, default=None)
    parser.add_argument('--test-data', type=str, default=None)
    parser.add_argument('--image-path-from', type=str, default=None)
    parser.add_argument('--image-path-to', type=str, default=None)
    parser.add_argument('--ablation', type=str,
                        choices=[None, 'language_only', 'visual_only'], default=None)

    args = parser.parse_args()

    log_dir = Path(__file__).parent / 'logs'
    setup_logging(log_dir=log_dir)
    logger = get_logger('dual_momemtum')

    try:
        logger.info("=" * 70)
        logger.info("Dual Momemtum search starting")
        logger.info("=" * 70)

        config = get_config()

        if args.iterations:
            config['num_iterations'] = args.iterations
        if args.api_key:
            config['meta_llm_api_key'] = args.api_key
        if args.output_dir:
            config['output_dir'] = args.output_dir
        if args.model_path:
            config['target_model_path'] = args.model_path
        if args.ablation:
            config['ablation_mode'] = args.ablation
        if args.train_data:
            config['train_data_path'] = args.train_data
        if args.dev_data:
            config['dev_data_path'] = args.dev_data
        if args.test_data:
            config['test_data_path'] = args.test_data
        if args.image_path_from:
            config['image_path_replacement']['from'] = args.image_path_from
        if args.image_path_to:
            config['image_path_replacement']['to'] = args.image_path_to

        # max_patience has no effect beyond num_iterations, cap it automatically
        config['max_patience'] = min(
            config.get('max_patience', 20),
            config['num_iterations']
        )

        if not validate_config(config):
            logger.error("Config validation failed")
            return 1

        logger.info(f"iterations={config['num_iterations']} | batch={config['batch_size']} | "
                    f"model={config['target_model_path']} | meta_llm={config['meta_llm_model']}")

        searcher = DualMomemtumSearcher(config)
        best_prompt, best_score = searcher.run_dual_momemtum(
            num_iterations=config['num_iterations']
        )

        prompt_path = searcher.prompt_manager.save_prompt(
            'best_dual_momemtum',
            best_prompt,
            metadata={
                'score': best_score,
                'method': 'dual_momemtum',
                'num_iterations': config['num_iterations'],
                'timestamp': datetime.now().isoformat(),
            },
        )

        logger.info(f"Done — best score: {best_score:.4f} | saved: {prompt_path}")
        return 0

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130

    except Exception as e:
        logger.error(f"Failed: {type(e).__name__}: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
