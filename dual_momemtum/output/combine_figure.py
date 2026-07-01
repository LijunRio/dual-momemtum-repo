import argparse
import random
import re
from pathlib import Path

from PIL import Image


DEFAULT_VIS_DIR = Path(
    "/u/home/lj0/Code/dual-momemtum-repo/dual_momemtum/output/"
    "qwen2.5-vl-7b/visualizations"
)
DEFAULT_OUTPUT = DEFAULT_VIS_DIR.parent / "failure_cases_7b.png"
DEFAULT_PDF_OUTPUT = DEFAULT_VIS_DIR.parent / "failure_cases_7b.pdf"

GRID_COLS = 3
GRID_ROWS =5
CELL_SIZE = 480
GAP = 6
RANDOM_SELECT = True
RANDOM_SEED = None  # set to an int, e.g. 42, to make the random selection reproducible
DEDUPLICATE_IMAGES = True
DUPLICATE_DISTANCE_THRESHOLD = 3.5  # lower is stricter; average RGB difference after resizing

FILENAME_RE = re.compile(
    r"viz_iter(?P<iteration>\d+)_vf(?P<vf>\d+)_error(?P<error>\d+)_.+\.(jpg|jpeg|png)$",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Combine visualization images into a grid.")
    parser.add_argument("--vis-dir", type=Path, default=DEFAULT_VIS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pdf-output", type=Path, default=DEFAULT_PDF_OUTPUT)
    parser.add_argument("--cols", type=int, default=GRID_COLS)
    parser.add_argument("--rows", type=int, default=GRID_ROWS)
    parser.add_argument("--cell-size", type=int, default=CELL_SIZE)
    parser.add_argument("--gap", type=int, default=GAP)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--no-dedup", action="store_true")
    parser.add_argument("--duplicate-threshold", type=float, default=DUPLICATE_DISTANCE_THRESHOLD)
    return parser.parse_args()


def collect_images(vis_dir):
    images = []
    for path in vis_dir.iterdir():
        if not path.is_file():
            continue
        match = FILENAME_RE.match(path.name)
        if not match:
            continue
        images.append({
            "path": path,
            "iteration": int(match.group("iteration")),
            "vf": int(match.group("vf")),
            "error": int(match.group("error")),
        })
    return sorted(images, key=lambda item: (item["iteration"], item["vf"], item["error"]))


def image_signature(path, size=64):
    with Image.open(path) as image:
        image = image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    return list(image.getdata())


def mean_abs_rgb_distance(left, right):
    total = 0
    count = min(len(left), len(right))
    if count == 0:
        return float("inf")

    for left_pixel, right_pixel in zip(left[:count], right[:count]):
        total += abs(left_pixel[0] - right_pixel[0])
        total += abs(left_pixel[1] - right_pixel[1])
        total += abs(left_pixel[2] - right_pixel[2])
    return total / (count * 3)


def deduplicate_images(images, duplicate_threshold=DUPLICATE_DISTANCE_THRESHOLD):
    unique_images = []
    unique_signatures = []
    skipped_duplicates = 0

    for item in images:
        try:
            item_signature = image_signature(item["path"])
        except Exception:
            unique_images.append(item)
            continue

        is_duplicate = any(
            mean_abs_rgb_distance(item_signature, unique_signature) <= duplicate_threshold
            for unique_signature in unique_signatures
        )
        if is_duplicate:
            skipped_duplicates += 1
            continue

        unique_images.append(item)
        unique_signatures.append(item_signature)

    return unique_images, skipped_duplicates


def select_images(images, limit, seed=RANDOM_SEED, deduplicate=DEDUPLICATE_IMAGES, duplicate_threshold=DUPLICATE_DISTANCE_THRESHOLD):
    candidates = list(images)
    skipped_duplicates = 0

    if deduplicate:
        candidates, skipped_duplicates = deduplicate_images(
            candidates, duplicate_threshold=duplicate_threshold
        )

    if RANDOM_SELECT:
        random.Random(seed).shuffle(candidates)

    selected = candidates[:limit]
    selected = sorted(selected, key=lambda item: (item["iteration"], item["vf"], item["error"]))
    return selected, skipped_duplicates

def resize_to_cell(image, size):
    image = image.convert("RGB")
    if image.size == (size, size):
        return image.copy()
    return image.resize((size, size), Image.Resampling.LANCZOS)


def make_grid(images, output, pdf_output=None, cols=GRID_COLS, rows=GRID_ROWS, cell_size=CELL_SIZE, gap=GAP, seed=RANDOM_SEED, deduplicate=DEDUPLICATE_IMAGES, duplicate_threshold=DUPLICATE_DISTANCE_THRESHOLD):
    selected, skipped_duplicates = select_images(
        images, cols * rows, seed=seed, deduplicate=deduplicate,
        duplicate_threshold=duplicate_threshold
    )
    if not selected:
        raise FileNotFoundError("No visualization images found.")

    width = cols * cell_size + (cols - 1) * gap
    height = rows * cell_size + (rows - 1) * gap
    canvas = Image.new("RGB", (width, height), (255, 255, 255))

    for idx, item in enumerate(selected):
        row = idx // cols
        col = idx % cols
        x = col * (cell_size + gap)
        y = row * (cell_size + gap)
        with Image.open(item["path"]) as image:
            canvas.paste(resize_to_cell(image, cell_size), (x, y))

    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG")
    if pdf_output:
        pdf_output.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(pdf_output, format="PDF", resolution=300.0)
    return selected, skipped_duplicates


def main():
    args = parse_args()
    images = collect_images(args.vis_dir)
    selected, skipped_duplicates = make_grid(
        images,
        args.output,
        pdf_output=args.pdf_output,
        cols=args.cols,
        rows=args.rows,
        cell_size=args.cell_size,
        gap=args.gap,
        seed=args.seed,
        deduplicate=not args.no_dedup,
        duplicate_threshold=args.duplicate_threshold,
    )
    print(f"Saved PNG: {args.output}")
    print(f"Saved PDF: {args.pdf_output}")
    print(f"Merged {len(selected)} / {len(images)} images")
    print(f"Skipped near-duplicates: {skipped_duplicates}")
    print("Selected:")
    for item in selected:
        print(f"  - {item['path'].name}")


if __name__ == "__main__":
    main()
