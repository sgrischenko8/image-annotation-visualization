"""Check the dataset: photos, YOLO annotation files and how well they match.

Usage:
    python check_dataset.py [--images DIR] [--labels DIR] [--classes FILE] [--strict]

Exit code: 0 = no errors, 1 = errors found (or warnings with --strict),
2 = folders/classes file could not be read.
"""

import argparse
import sys
from collections import Counter, defaultdict

from PIL import Image

from shared import (
    DRONE_CLASS_NAME,
    IMAGE_SUFFIXES,
    SKIP_CLASS_NAME,
    TOLERANCE,
    add_folder_args,
    box_to_corners,
    ensure_folder,
    find_duplicate_boxes,
    find_images,
    find_label_files,
    parse_annotation_line,
    read_class_names,
)

# Labeling rules, section 7: objects smaller than ~8x8 px are not labeled.
MIN_SIDE_PX = 8

# Heuristic: a drone wider/taller than this share of the photo is suspicious.
DRONE_MAX_SIDE = 0.25


def run_checks(image_dir, label_dir, classes):
    errors = []
    warnings = []
    per_class = Counter()
    annotation_total = 0
    empty_images = 0

    skip_ids = {i for i, n in classes.items() if n == SKIP_CLASS_NAME}
    drone_ids = {i for i, n in classes.items() if n == DRONE_CLASS_NAME}

    photos = find_images(image_dir)
    label_files = find_label_files(label_dir)

    # Files in the images folder that will never be checked.
    for path in sorted(image_dir.iterdir()):
        if (path.is_file() and not path.name.startswith(".")
                and path.suffix.lower() not in IMAGE_SUFFIXES):
            warnings.append(f"{path.name}: unsupported file in images folder (skipped)")

    # Two photos with the same base name would share one annotation file.
    names_by_stem = defaultdict(list)
    for photo in photos:
        names_by_stem[photo.stem].append(photo.name)
    for names in names_by_stem.values():
        if len(names) > 1:
            errors.append(
                "photos share the same base name and would use the same annotation file: "
                + ", ".join(names)
            )

    # Annotation files without a photo.
    photo_stems = {p.stem for p in photos}
    for label_file in label_files:
        if label_file.stem not in photo_stems:
            warnings.append(f"{label_file.name}: annotation file has no matching photo")

    for photo in photos:
        # ---- photo ----
        try:
            with Image.open(photo) as img:
                photo_width, photo_height = img.size
        except Exception as e:
            errors.append(f"{photo.name}: cannot open image ({e})")
            continue

        if photo_width <= 0 or photo_height <= 0:
            errors.append(f"{photo.name}: invalid image dimensions")
            continue

        # ---- annotation file ----
        label_file = label_dir / f"{photo.stem}.txt"
        if not label_file.exists():
            errors.append(f"{photo.name}: annotation file is missing")
            continue

        try:
            rows = label_file.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as e:
            errors.append(f"{label_file.name}: cannot read annotation ({e})")
            continue

        valid_entries = []  # (line_number, class_id, box) of valid boxes

        for line_number, row in enumerate(rows, start=1):
            if not row.strip():
                continue

            where = f"{label_file.name}:{line_number}"

            try:
                class_id, box = parse_annotation_line(row)
            except ValueError as e:
                errors.append(f"{where}: {e}")
                continue

            annotation_total += 1

            if class_id not in classes:
                errors.append(f"{where}: unknown class ID {class_id}")
                continue

            x_center, y_center, width, height = box
            row_ok = True

            for name, value in (
                ("x_center", x_center), ("y_center", y_center),
                ("width", width), ("height", height),
            ):
                if not -TOLERANCE <= value <= 1 + TOLERANCE:
                    errors.append(f"{where}: {name}={value} outside [0, 1]")
                    row_ok = False

            if width <= 0 or height <= 0:
                errors.append(f"{where}: bbox has zero or negative size")
                continue

            if row_ok:
                x1, y1, x2, y2 = box_to_corners(box)
                if x1 < -TOLERANCE or y1 < -TOLERANCE or x2 > 1 + TOLERANCE or y2 > 1 + TOLERANCE:
                    errors.append(f"{where}: bbox goes outside image")
                    row_ok = False

            if not row_ok:
                continue

            class_name = classes[class_id]
            per_class[class_name] += 1

            # Size checks (heuristics -> warnings). Skip-class regions may be tiny.
            if class_id not in skip_ids:
                width_px = width * photo_width
                height_px = height * photo_height
                if width_px < MIN_SIDE_PX or height_px < MIN_SIDE_PX:
                    warnings.append(
                        f"{where}: {class_name} box is {width_px:.1f}x{height_px:.1f} px "
                        f"(smaller than {MIN_SIDE_PX} px)"
                    )
            if class_id in drone_ids and (width > DRONE_MAX_SIDE or height > DRONE_MAX_SIDE):
                warnings.append(
                    f"{where}: drone box is very large "
                    f"({width:.2f}x{height:.2f} of the image)"
                )

            valid_entries.append((line_number, class_id, box))

        if not valid_entries:
            empty_images += 1

        # ---- duplicates ----
        for kept_line, dup_line, class_id, iou in find_duplicate_boxes(valid_entries):
            warnings.append(
                f"{label_file.name}: duplicate annotations at lines "
                f"{kept_line} and {dup_line} "
                f"(class={classes[class_id]}, IoU={iou:.3f})"
            )

    return {
        "photos": len(photos),
        "annotations": annotation_total,
        "per_class": per_class,
        "empty_images": empty_images,
        "errors": errors,
        "warnings": warnings,
    }


def show_report(outcome, classes):
    errors = outcome["errors"]
    warnings = outcome["warnings"]

    print("=" * 60)
    print("DATASET CHECK")
    print("=" * 60)
    print(f"Images checked:          {outcome['photos']}")
    print(f"Annotations checked:     {outcome['annotations']}")
    print(f"Images without objects:  {outcome['empty_images']}")
    print("Boxes per class:")
    for name in classes.values():
        print(f"  {name:<10} {outcome['per_class'].get(name, 0)}")
    print(f"Errors:                  {len(errors)}")
    print(f"Warnings:                {len(warnings)}")

    if errors:
        print("\nERRORS:")
        for error in errors:
            print(f"  - {error}")

    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")

    if not errors and not warnings:
        print("\n✓ Dataset passed the check!")
    elif not errors:
        print("\n✓ No critical errors found. Review the warnings.")
    else:
        print("\n✗ Dataset contains errors. Review the report above.")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_folder_args(parser)
    parser.add_argument(
        "--strict", action="store_true",
        help="exit with code 1 if there are warnings too",
    )
    args = parser.parse_args()

    try:
        ensure_folder(args.images, "Image")
        ensure_folder(args.labels, "Label")
        classes = read_class_names(args.classes)
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        return 2

    if not find_images(args.images):
        print("ERROR: No images found.")
        return 2

    outcome = run_checks(args.images, args.labels, classes)
    show_report(outcome, classes)

    if outcome["errors"] or (args.strict and outcome["warnings"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())