"""Overlay YOLO annotations on their matching photos for visual review."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from shared import (
    add_folder_args,
    ensure_folder,
    find_images,
    parse_annotation_line,
    read_class_names,
)


# The palette repeats if a project has more classes than colours below.
PALETTE = (
    (230, 57, 70),    # red
    (29, 185, 84),    # green
    (0, 123, 255),    # blue
    (255, 159, 28),   # orange
    (142, 68, 173),   # purple
    (0, 172, 193),    # turquoise
)


def target_path(out_dir: Path, photo_path: Path) -> Path:
    """Use PNG for formats Pillow cannot reliably save after drawing."""
    if photo_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return out_dir / photo_path.name
    return out_dir / f"{photo_path.stem}.png"


def paint_box(canvas: ImageDraw.ImageDraw, box, photo_size, color, caption: str) -> None:
    """Convert a normalized YOLO box to pixels and draw it with a class caption."""
    x_center, y_center, width, height = box
    photo_width, photo_height = photo_size
    left = round((x_center - width / 2) * photo_width)
    top = round((y_center - height / 2) * photo_height)
    right = round((x_center + width / 2) * photo_width)
    bottom = round((y_center + height / 2) * photo_height)

    stroke = max(2, round(min(photo_size) / 400))
    canvas.rectangle((left, top, right, bottom), outline=color, width=stroke)
    caption_area = canvas.textbbox((left, top), caption)
    caption_width = caption_area[2] - caption_area[0] + 6
    caption_height = caption_area[3] - caption_area[1] + 4
    caption_top = top - caption_height if top >= caption_height else top
    canvas.rectangle((left, caption_top, left + caption_width, caption_top + caption_height), fill=color)
    canvas.text((left + 3, caption_top + 2), caption, fill="white")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Draw colored bounding boxes from YOLO annotations onto photos."
    )
    add_folder_args(parser)
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).resolve().parent / "preview_images",
        help="folder for the preview copies (default: ./preview_images)",
    )
    args = parser.parse_args()

    ensure_folder(args.images, "Images")
    ensure_folder(args.labels, "Labels")
    classes = read_class_names(args.classes)
    args.out.mkdir(parents=True, exist_ok=True)

    done = 0
    skipped = 0
    for photo_path in find_images(args.images):
        label_file = args.labels / f"{photo_path.stem}.txt"
        if not label_file.is_file():
            print(f"SKIP {photo_path.name}: no matching label file")
            skipped += 1
            continue

        try:
            with Image.open(photo_path) as source:
                picture = source.convert("RGB")
        except OSError as error:
            print(f"SKIP {photo_path.name}: cannot open image ({error})")
            skipped += 1
            continue

        canvas = ImageDraw.Draw(picture)
        for line_number, row in enumerate(label_file.read_text(encoding="utf-8").splitlines(), 1):
            if not row.strip():
                continue
            try:
                class_id, box = parse_annotation_line(row)
            except ValueError as error:
                print(f"WARNING {label_file.name}:{line_number}: {error}; skipped")
                continue
            if class_id not in classes:
                print(f"WARNING {label_file.name}:{line_number}: unknown class ID {class_id}; skipped")
                continue

            paint_box(canvas, box, picture.size, PALETTE[class_id % len(PALETTE)], classes[class_id])

        destination = target_path(args.out, photo_path)
        try:
            picture.save(destination, quality=95)
        except OSError as error:
            print(f"SKIP {photo_path.name}: cannot save output ({error})")
            skipped += 1
            continue
        print(f"SAVED {destination}")
        done += 1

    print(f"Done: {done} image(s) saved, {skipped} image(s) skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())