"""Shared helpers for the drone dataset tools (YOLO annotation files)."""

import argparse
import math
import os
import shutil
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Slack for floating-point noise when comparing coordinates with 0 and 1.
TOLERANCE = 1e-6

# Two boxes of the same class whose IoU is above this value count as duplicates.
DUPLICATE_IOU_LIMIT = 0.95

DRONE_CLASS_NAME = "drone"
SKIP_CLASS_NAME = "ignore"  # optional: only used if such a class is added to class_names.txt


def add_folder_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register --images / --labels / --classes options (defaults: next to the scripts)."""
    parser.add_argument(
        "--images", type=Path, default=ROOT_DIR / "images",
        help="folder with photos (default: ./images)",
    )
    parser.add_argument(
        "--labels", type=Path, default=ROOT_DIR / "labels",
        help="folder with YOLO .txt annotation files (default: ./labels)",
    )
    parser.add_argument(
        "--classes", type=Path, default=ROOT_DIR / "class_names.txt",
        help="file with class names, one per line (default: ./class_names.txt)",
    )
    return parser


def read_class_names(path: Path) -> dict:
    """Load the class list. The line index (starting at 0) is the class ID."""
    if not path.is_file():
        raise FileNotFoundError(f"Classes file not found: {path}")
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    while names and not names[-1]:
        names.pop()
    if not names:
        raise ValueError(f"Classes file is empty: {path}")
    if any(not name for name in names):
        raise ValueError(f"Classes file contains an empty line in the middle: {path}")
    return dict(enumerate(names))


def ensure_folder(path: Path, what: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{what} folder not found: {path}")


def find_images(image_dir: Path) -> list:
    return sorted(
        (p for p in image_dir.iterdir()
         if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: p.name.lower(),
    )


def find_label_files(label_dir: Path) -> list:
    return sorted(
        (p for p in label_dir.glob("*.txt") if p.is_file()),
        key=lambda p: p.name.lower(),
    )


def parse_annotation_line(line: str):
    """Parse 'class x_center y_center width height'.

    Returns (class_id, (x_center, y_center, width, height)).
    Raises ValueError with a readable message when the line is malformed.
    """
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"expected 5 values, got {len(parts)}")
    try:
        class_id = int(parts[0])
    except ValueError:
        raise ValueError(f"class ID {parts[0]!r} is not an integer") from None
    try:
        box = tuple(float(p) for p in parts[1:])
    except ValueError:
        raise ValueError("invalid numeric value") from None
    if not all(math.isfinite(v) for v in box):
        raise ValueError("non-finite numeric value")
    return class_id, box


def format_annotation_line(class_id: int, box) -> str:
    """Build an annotation line with fixed precision (never scientific notation)."""
    return f"{class_id} " + " ".join(f"{v:.6f}" for v in box)


def box_to_corners(box):
    """(x_center, y_center, w, h) -> (x1, y1, x2, y2)."""
    x, y, w, h = box
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2


def compute_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_to_corners(box_a)
    bx1, by1, bx2, by2 = box_to_corners(box_b)

    overlap_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    overlap_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    overlap = overlap_w * overlap_h

    total = box_a[2] * box_a[3] + box_b[2] * box_b[3] - overlap
    if total <= 0:
        return 0.0
    return overlap / total


def find_duplicate_boxes(entries, limit: float = DUPLICATE_IOU_LIMIT):
    """Locate duplicate boxes of the same class.

    entries: list of (line_number, class_id, box).
    Returns a list of (kept_line, duplicate_line, class_id, iou).
    The earliest box is kept; each later box that overlaps a kept box
    of the same class with IoU > limit is reported as a duplicate.
    """
    survivors = []
    found = []
    for line_number, class_id, box in entries:
        for kept_line, kept_class, kept_box in survivors:
            if kept_class != class_id:
                continue
            iou = compute_iou(kept_box, box)
            if iou > limit:
                found.append((kept_line, line_number, class_id, iou))
                break
        else:
            survivors.append((line_number, class_id, box))
    return found


def save_text_safely(path: Path, text: str) -> None:
    """Write a file so that a crash never leaves it half-written."""
    fd, temp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        shutil.copymode(path, temp_path)
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise