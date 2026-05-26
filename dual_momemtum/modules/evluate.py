__all__ = [
    'calculate_iou',
    'parse_boxes_from_output',
    'parse_ground_truth',
    'calculate_map',
    'calculate_map50',
    'calculate_map30',
]

import json
import re
from typing import List


def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def parse_boxes_from_output(text):
    """Extract all bbox_2d and label entries from model output."""
    if not text or text.strip() == "":
        return [{"label": "None", "bbox_2d": [0, 0, 0, 0]}]
    results = []

    cleaned_text = re.sub(r'<[^>]*>', '', text)
    cleaned_text = re.sub(r'```json\s*', '', cleaned_text)
    cleaned_text = re.sub(r'```\s*$', '', cleaned_text).strip()

    try:
        json_data = json.loads(cleaned_text)
        if isinstance(json_data, list):
            for item in json_data:
                if isinstance(item, dict) and "bbox_2d" in item and "label" in item:
                    results.append({"label": item["label"], "bbox_2d": item["bbox_2d"]})
        if results:
            return results
    except Exception:
        try:
            json_data = json.loads(cleaned_text.replace("'", '"'))
            if isinstance(json_data, list):
                for item in json_data:
                    if isinstance(item, dict) and "bbox_2d" in item and "label" in item:
                        results.append({"label": item["label"], "bbox_2d": item["bbox_2d"]})
            if results:
                return results
        except Exception:
            pass

    pattern = r'"bbox_2d"\s*:\s*\[\s*([^\]]+)\s*\][^}]*"label"\s*:\s*"([^"]+)"'
    for bbox_str, label in re.findall(pattern, text):
        try:
            bbox_2d = [int(x.strip()) for x in bbox_str.split(',')]
            if len(bbox_2d) == 4:
                results.append({"label": label, "bbox_2d": bbox_2d})
        except Exception:
            continue

    pattern2 = r'"label"\s*:\s*"([^"]+)"[^}]*"bbox_2d"\s*:\s*\[\s*([^\]]+)\s*\]'
    for label, bbox_str in re.findall(pattern2, text):
        try:
            bbox_2d = [int(x.strip()) for x in bbox_str.split(',')]
            if len(bbox_2d) == 4:
                results.append({"label": label, "bbox_2d": bbox_2d})
        except Exception:
            continue

    return results if results else [{"label": "None", "bbox_2d": [0, 0, 0, 0]}]


def parse_ground_truth(label_str: str) -> List[List[int]]:
    try:
        boxes = json.loads(label_str)
        if isinstance(boxes, list) and len(boxes) > 0:
            return boxes
        return []
    except Exception:
        return []


def calculate_map(predictions_list: List[List[List[int]]],
                  ground_truths_list: List[List[List[int]]],
                  iou_threshold: float = 0.5) -> float:
    if not predictions_list or not ground_truths_list:
        return 0.0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for predictions, ground_truths in zip(predictions_list, ground_truths_list):
        if len(ground_truths) == 0:
            total_fp += len(predictions)
            continue

        if len(predictions) == 0:
            total_fn += len(ground_truths)
            continue

        matched_gts = set()
        for pred in predictions:
            best_iou = 0.0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx in matched_gts:
                    continue
                iou = calculate_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                total_tp += 1
                matched_gts.add(best_gt_idx)
            else:
                total_fp += 1

        total_fn += len(ground_truths) - len(matched_gts)

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)


def calculate_map50(predictions_list: List[List[List[int]]],
                    ground_truths_list: List[List[List[int]]]) -> float:
    return calculate_map(predictions_list, ground_truths_list, iou_threshold=0.5)


def calculate_map30(predictions_list: List[List[List[int]]],
                    ground_truths_list: List[List[List[int]]]) -> float:
    return calculate_map(predictions_list, ground_truths_list, iou_threshold=0.3)
