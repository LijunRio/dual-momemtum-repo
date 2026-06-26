import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from modules.evluate import calculate_iou, calculate_map
from modules.utils import get_image_size
from modules.logger import get_logger

logger = get_logger(__name__)


class Evaluator:

    def __init__(self):
        self.image_size_cache = {}

    def extract_ground_truth(self, sample: Dict) -> List[List[int]]:
        try:
            for msg in sample.get('messages', []):
                if msg.get('role') == 'assistant':
                    content = msg.get('content', '')
                    if isinstance(content, str):
                        try:
                            data = json.loads(content)
                            if isinstance(data, list):
                                return data
                        except Exception:
                            pass
            if 'gt' in sample:
                gt = sample['gt']
                if isinstance(gt, str):
                    try:
                        gt = json.loads(gt)
                    except Exception:
                        pass
                if isinstance(gt, list):
                    return gt
            return []
        except Exception as e:
            print(f"⚠️ Failed to extract ground truth: {e}")
            return []

    def normalize_bbox(self, bbox: List[int], image_width: int = 476,
                       image_height: int = 476) -> List[int]:
        x1, y1, x2, y2 = bbox
        return [
            int(x1 * image_width / 1000),
            int(y1 * image_height / 1000),
            int(x2 * image_width / 1000),
            int(y2 * image_height / 1000),
        ]

    def get_image_size(self, image_path: str, path_from: Optional[str] = None,
                       path_to: Optional[str] = None) -> Tuple[int, int]:
        if image_path in self.image_size_cache:
            return self.image_size_cache[image_path]
        width, height = get_image_size(image_path, path_from, path_to)
        self.image_size_cache[image_path] = (width, height)
        return width, height

    def evaluate_batch(self, infer_results: List[Dict], coord_mode: str = 'pixel',
                       path_from: str = None, path_to: str = None) -> Tuple[List[float], List[Dict]]:
        scores = []
        errors = []
        no_target_count = 0
        no_gt_count = 0
        zero_score_count = 0

        for result in infer_results:
            sample = result['sample']
            pred_boxes_raw = result['parsed_boxes']
            gt_boxes_raw = self.extract_ground_truth(sample)

            pred_boxes = []
            if isinstance(pred_boxes_raw, list):
                for item in pred_boxes_raw:
                    if isinstance(item, dict) and 'bbox_2d' in item:
                        bbox = item['bbox_2d']
                        if isinstance(bbox, list) and len(bbox) == 4:
                            try:
                                parsed_bbox = [float(x) for x in bbox]
                            except (ValueError, TypeError):
                                continue
                            if self._is_valid_box(parsed_bbox):
                                pred_boxes.append(parsed_bbox)
                    elif isinstance(item, list) and len(item) == 4:
                        try:
                            parsed_bbox = [float(x) for x in item]
                        except (ValueError, TypeError):
                            continue
                        if self._is_valid_box(parsed_bbox):
                            pred_boxes.append(parsed_bbox)

            gt_boxes = []
            if isinstance(gt_boxes_raw, list):
                for bbox in gt_boxes_raw:
                    if isinstance(bbox, list) and len(bbox) == 4:
                        try:
                            parsed_bbox = [float(x) for x in bbox]
                        except (ValueError, TypeError):
                            continue
                        if self._is_valid_box(parsed_bbox):
                            gt_boxes.append(parsed_bbox)

            if len(pred_boxes) == 0 and len(gt_boxes) == 0:
                no_target_count += 1
            elif len(gt_boxes) == 0:
                no_gt_count += 1

            if coord_mode == 'normalize' and pred_boxes:
                img_w, img_h = self.get_image_size(result['image_path'], path_from, path_to)
                pred_boxes = [self.normalize_bbox(b, img_w, img_h) for b in pred_boxes]

            try:
                score = calculate_map([pred_boxes], [gt_boxes], iou_threshold=0.5)
            except Exception:
                score = 0.0

            if score == 0.0:
                zero_score_count += 1

            scores.append(score)

            logger.debug(f"sample {result['sample_idx']}: {result['image_path']}")
            logger.debug(f"  pred={len(pred_boxes)}, gt={len(gt_boxes)}, mAP@0.5={score:.4f}")

            if score < 0.2:
                best_iou = self._best_iou(pred_boxes, gt_boxes)
                errors.append({
                    'sample_idx': result['sample_idx'],
                    'image_path': result['image_path'],
                    'sample': result['sample'],
                    'pred_boxes': pred_boxes,
                    'gt_boxes': gt_boxes,
                    'score': score,
                    'best_iou': best_iou,
                    'pred_count': len(pred_boxes),
                    'gt_count': len(gt_boxes),
                    'error_type': self._classify_error(pred_boxes, gt_boxes, best_iou),
                })

        batch_size = len(infer_results)
        if batch_size > 0:
            logger.debug(f"Batch: total={batch_size}, no_target={no_target_count}, "
                         f"no_gt={no_gt_count}, zero_score={zero_score_count} "
                         f"({100*zero_score_count/batch_size:.1f}%), "
                         f"low_score={len(errors)} ({100*len(errors)/batch_size:.1f}%)")

        return scores, errors

    @staticmethod
    def _is_valid_box(box: List) -> bool:
        if len(box) != 4:
            return False
        x1, y1, x2, y2 = box
        return x2 > x1 and y2 > y1

    @staticmethod
    def _best_iou(pred_boxes: List, gt_boxes: List) -> float:
        if not pred_boxes or not gt_boxes:
            return 0.0

        best_iou = 0.0
        for pred_box in pred_boxes:
            for gt_box in gt_boxes:
                try:
                    best_iou = max(best_iou, calculate_iou(pred_box, gt_box))
                except Exception:
                    continue
        return best_iou

    @staticmethod
    def _classify_error(pred_boxes: List, gt_boxes: List, best_iou: float = None) -> str:
        pred_count = len(pred_boxes)
        gt_count = len(gt_boxes)

        if pred_count == 0 and gt_count == 0:
            return "correct_no_target"
        if pred_count == 0:
            return "missed_target"
        if gt_count == 0:
            return "false_positive"

        if best_iou is None:
            best_iou = Evaluator._best_iou(pred_boxes, gt_boxes)

        if best_iou == 0.0:
            return "wrong_region"
        if best_iou < 0.3:
            return "poor_overlap"
        if best_iou < 0.5:
            return "near_miss"
        if pred_count > gt_count:
            return "over_detection"
        if pred_count < gt_count:
            return "under_detection"
        return "low_confidence_or_label_error"

    def select_worst_cases(self, errors: List[Dict], k: int = 20) -> List[Dict]:
        if not errors:
            return []
        return sorted(errors, key=lambda x: x['score'])[:min(k, len(errors))]

    def summarize_errors(self, errors: List[Dict]) -> Dict:
        if not errors:
            return {'total_errors': 0, 'error_types': {}}
        error_types = defaultdict(int)
        for error in errors:
            error_types[error.get('error_type', 'unknown')] += 1
        return {'total_errors': len(errors), 'error_types': dict(error_types)}
