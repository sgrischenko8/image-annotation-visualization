"""Clip YOLO boxes that stick out of the photo to the photo borders.

Usage:
    python clip_boxes.py [--labels DIR] [--dry-run] [--max-overshoot 0.02]

- Small overshoots (up to --max-overshoot, as a share of the photo side) are clipped.
- Boxes that lie completely outside the photo are removed.
- Large overshoots are NOT changed: they are most likely annotation mistakes
  and are listed for manual review.
- All other lines in a file are kept exactly as they are.

Exit code: 0 = nothing left to review, 1 = some boxes need manual review.
"""

import argparse
import sys
from pathlib import Path

from shared import (
    ROOT_DIR,
    TOLERANCE,
    box_to_corners,
    ensure_folder,
    find_label_files,
    format_annotation_line,
    parse_annotation_line,
    save_text_safely,
)

DEFAULT_MAX_OVERSHOOT = 0.02


def clip_file(label_file, max_overshoot):
    """Return (new_text or None, clipped_count, removed_count, review_notes)."""
    rows = label_file.read_text(encoding="utf-8").splitlines()
    output_rows = []
    clipped_count = 0
    removed_count = 0
    review_notes = []

    for line_number, row in enumerate(rows, start=1):
        if not row.strip():
            output_rows.append(row)
            continue

        try:
            class_id, box = parse_annotation_line(row)
        except ValueError:
            output_rows.append(row)  # check_dataset.py reports malformed lines
            continue

        if box[2] <= 0 or box[3] <= 0:
            output_rows.append(row)  # check_dataset.py reports invalid sizes
            continue

        x1, y1, x2, y2 = box_to_corners(box)
        overshoot = max(-x1, -y1, x2 - 1, y2 - 1)

        if overshoot <= TOLERANCE:
            output_rows.append(row)
            continue

        nx1, ny1, nx2, ny2 = (min(1.0, max(0.0, v)) for v in (x1, y1, x2, y2))
        new_width = nx2 - nx1
        new_height = ny2 - ny1

        if new_width <= TOLERANCE or new_height <= TOLERANCE:
            removed_count += 1
            continue  # the box is completely outside the photo: drop it

        if overshoot > max_overshoot:
            review_notes.append(
                f"{label_file.name}:{line_number}: box sticks out by "
                f"{overshoot:.3f} of the image size (> {max_overshoot}); check manually"
            )
            output_rows.append(row)
            continue

        new_box = ((nx1 + nx2) / 2, (ny1 + ny2) / 2, new_width, new_height)
        output_rows.append(format_annotation_line(class_id, new_box))
        clipped_count += 1

    if not clipped_count and not removed_count:
        return None, 0, 0, review_notes

    text = "\n".join(output_rows) + ("\n" if output_rows else "")
    return text, clipped_count, removed_count, review_notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--labels", type=Path, default=ROOT_DIR / "labels",
                        help="folder with YOLO .txt annotation files (default: ./labels)")
    parser.add_argument("--dry-run", action="store_true",
                        help="show what would change without modifying files")
    parser.add_argument("--max-overshoot", type=float, default=DEFAULT_MAX_OVERSHOOT,
                        help=f"largest overshoot (share of the photo) that is "
                             f"clipped automatically (default: {DEFAULT_MAX_OVERSHOOT})")
    args = parser.parse_args()

    try:
        ensure_folder(args.labels, "Label")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 2

    changed_files = 0
    total_clipped = 0
    total_removed = 0
    all_notes = []

    for label_file in find_label_files(args.labels):
        try:
            text, clipped_count, removed_count, notes = clip_file(label_file, args.max_overshoot)
        except (OSError, UnicodeDecodeError) as e:
            print(f"ERROR: cannot read {label_file.name}: {e}")
            continue

        all_notes.extend(notes)
        if text is None:
            continue

        changed_files += 1
        total_clipped += clipped_count
        total_removed += removed_count
        print(f"{'Would fix' if args.dry_run else 'Fixed'}: {label_file.name} "
              f"(clipped: {clipped_count}, removed: {removed_count})")
        if not args.dry_run:
            save_text_safely(label_file, text)

    print()
    print("=" * 50)
    print("BOX CLIPPING RESULT" + (" (dry run)" if args.dry_run else ""))
    print("=" * 50)
    print(f"Files modified:                {changed_files}")
    print(f"Boxes clipped:                 {total_clipped}")
    print(f"Boxes removed (fully outside): {total_removed}")
    print(f"Boxes to review manually:      {len(all_notes)}")
    for note in all_notes:
        print(f"  - {note}")
    print("=" * 50)

    return 1 if all_notes else 0


if __name__ == "__main__":
    sys.exit(main())