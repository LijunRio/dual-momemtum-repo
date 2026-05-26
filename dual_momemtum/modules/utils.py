import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
from PIL import Image


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, filepath: Path, indent: int = 2) -> None:
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(filepath: Path) -> Any:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_jsonl(records: list, filepath: Path) -> None:
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    with open(filepath, 'a', encoding='utf-8') as f:
        for record in records:
            json.dump(record, f, ensure_ascii=False)
            f.write('\n')


def append_jsonl(record: Dict, filepath: Path) -> None:
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    with open(filepath, 'a', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False)
        f.write('\n')


def read_text_file(filepath: Path) -> str:
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()


def write_text_file(content: str, filepath: Path) -> None:
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)


def get_image_size(image_path: str, path_from: Optional[str] = None,
                   path_to: Optional[str] = None) -> tuple:
    if path_from and path_to and path_from in image_path:
        image_path = image_path.replace(path_from, path_to)
    try:
        img = Image.open(image_path)
        return img.width, img.height
    except Exception as e:
        logging.warning(f"Cannot get image size {image_path}: {e}")
        return 0, 0


def format_score_diff(score: float, base_score: float) -> str:
    diff = score - base_score
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.4f}"
