"""Drop duplicate YOLO boxes (same class, IoU above a limit).

Usage:
    python remove_duplicates.py [--labels DIR] [--iou-threshold 0.95] [--dry-run]

The first box of a duplicate group is kept, later ones are dropped.
All other lines in a file are kept exactly as they are.
"""

import argparse
import sys
from pathlib import Path

from shared import (
    DUPLICATE_IOU_LIMIT,
    ROOT_DIR,
    ensure_folder,
    find_duplicate_boxes,
    find_label_files,
    parse_annotation_line,
    save_text_safely,
)


def filter_file(label_file, iou_limit):
    """Return (new_text or None, [(kept_line, dropped_line, class_id, iou)])."""
    rows = label_file.read_text(encoding="utf-8").splitlines()

    entries = []
    for line_number, row in enumerate(rows, start=1):
        if not row.strip():
            continue
        try:
            class_id, box = parse_annotation_line(row)
        except ValueError:
            continue  # check_dataset.py reports malformed lines
        if box[2] <= 0 or box[3] <= 0:
            continue
        entries.append((line_number, class_id, box))

    found = find_duplicate_boxes(entries, iou_limit)
    if not found:
        return None, []

    to_drop = {dup_line for _, dup_line, _, _ in found}
    remaining = [r for i, r in enumerate(rows, start=1) if i not in to_drop]
    text = "\n".join(remaining) + ("\n" if remaining else "")
    return text, found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--labels", type=Path, default=ROOT_DIR / "labels",
                        help="folder with YOLO .txt annotation files (default: ./labels)")
    parser.add_argument("--iou-threshold", type=float, default=DUPLICATE_IOU_LIMIT,
                        help=f"IoU above which boxes are duplicates "
                             f"(default: {DUPLICATE_IOU_LIMIT})")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would be dropped without modifying files")
    args = parser.parse_args()

    try:
        ensure_folder(args.labels, "Label")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 2

    changed_files = 0
    dropped_total = 0

    for label_file in find_label_files(args.labels):
        try:
            text, found = filter_file(label_file, args.iou_threshold)
        except (OSError, UnicodeDecodeError) as e:
            print(f"ERROR: cannot read {label_file.name}: {e}")
            continue

        if text is None:
            continue

        changed_files += 1
        dropped_total += len(found)
        for kept, dropped, class_id, iou in found:
            print(f"{label_file.name}: line {dropped} duplicates line {kept} "
                  f"(class {class_id}, IoU={iou:.3f})")
        if not args.dry_run:
            save_text_safely(label_file, text)

    print()
    print("=" * 50)
    print("DUPLICATE REMOVAL RESULT" + (" (dry run)" if args.dry_run else ""))
    print("=" * 50)
    print(f"Files modified:  {changed_files}")
    print(f"Boxes removed:   {dropped_total}")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())