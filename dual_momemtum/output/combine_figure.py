import argparse
import re
from pathlib import Path

from PIL import Image


DEFAULT_VIS_DIR = Path(
    "/u/home/lj0/Code/dual-momemtum-repo/dual_momemtum/output/"
    "qwen2.5-vl-3b/visualizations"
)
DEFAULT_OUTPUT = DEFAULT_VIS_DIR.parent / "optimizer_failure_cases.png"
DEFAULT_PDF_OUTPUT = DEFAULT_VIS_DIR.parent / "optimizer_failure_cases.pdf"

GRID_COLS = 3
GRID_ROWS = 5
CELL_SIZE = 480
GAP = 6

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


def select_diverse_images(images, limit):
    by_iteration = {}
    for item in images:
        by_iteration.setdefault(item["iteration"], []).append(item)

    selected = []
    iteration_order = sorted(by_iteration)
    while len(selected) < limit:
        added = False
        for iteration in iteration_order:
            candidates = by_iteration[iteration]
            if not candidates:
                continue
            selected.append(candidates.pop(0))
            added = True
            if len(selected) >= limit:
                break
        if not added:
            break

    return selected


def resize_to_cell(image, size):
    image = image.convert("RGB")
    if image.size == (size, size):
        return image.copy()
    return image.resize((size, size), Image.Resampling.LANCZOS)


def make_grid(images, output, pdf_output=None, cols=GRID_COLS, rows=GRID_ROWS, cell_size=CELL_SIZE, gap=GAP):
    selected = select_diverse_images(images, cols * rows)
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
    return selected


def main():
    args = parse_args()
    images = collect_images(args.vis_dir)
    selected = make_grid(
        images,
        args.output,
        pdf_output=args.pdf_output,
        cols=args.cols,
        rows=args.rows,
        cell_size=args.cell_size,
        gap=args.gap,
    )
    print(f"Saved PNG: {args.output}")
    print(f"Saved PDF: {args.pdf_output}")
    print(f"Merged {len(selected)} / {len(images)} images")
    print("Selected:")
    for item in selected:
        print(f"  - {item['path'].name}")


if __name__ == "__main__":
    main()
