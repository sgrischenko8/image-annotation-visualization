"""Move photos that have no annotation file out of the images folder.

Usage:
    python move_unlabeled.py            # dry run: only shows what would happen
    python move_unlabeled.py --apply    # really moves the photos

Photos are MOVED to a separate folder (default: ./unlabeled_images), not deleted,
so the operation can be undone. Annotation files without a photo are only reported.

Note: a photo with an EMPTY annotation file is kept (it means "no objects").
Make sure your annotation tool exports an empty .txt file for such photos,
otherwise they will be treated as unlabeled.
"""

import argparse
import shutil
import sys
from pathlib import Path

from shared import (
    ROOT_DIR,
    add_folder_args,
    ensure_folder,
    find_images,
    find_label_files,
)


def free_destination(folder: Path, name: str) -> Path:
    target = folder / name
    counter = 1
    while target.exists():
        target = folder / f"{Path(name).stem}_{counter}{Path(name).suffix}"
        counter += 1
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_folder_args(parser)
    parser.add_argument("--apply", action="store_true",
                        help="actually move the photos (default is a dry run)")
    parser.add_argument("--target-dir", type=Path,
                        default=ROOT_DIR / "unlabeled_images",
                        help="where to move photos (default: ./unlabeled_images)")
    args = parser.parse_args()

    try:
        ensure_folder(args.images, "Image")
        ensure_folder(args.labels, "Label")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return 2

    photos = find_images(args.images)
    label_files = find_label_files(args.labels)
    label_stems = {p.stem for p in label_files}
    photo_stems = {p.stem for p in photos}

    to_move = [p for p in photos if p.stem not in label_stems]
    staying = [p for p in photos if p.stem in label_stems]
    orphan_labels = [p for p in label_files if p.stem not in photo_stems]

    # Safety net: a wrong --labels path would otherwise move everything.
    if photos and not staying:
        print("ERROR: none of the photos has an annotation file. "
              "Wrong labels folder? Nothing was moved.")
        return 2

    print("=" * 50)
    print("CLEANUP RESULT" + ("" if args.apply else " (dry run)"))
    print("=" * 50)
    print(f"Images found:       {len(photos)}")
    print(f"Label files found:  {len(label_files)}")
    print(f"Images kept:        {len(staying)}")
    print(f"Images to remove:   {len(to_move)}")

    if to_move:
        print("\nImages without a label file:")
        for path in to_move:
            print(f"  - {path.name}")

        if args.apply:
            args.target_dir.mkdir(parents=True, exist_ok=True)
            for path in to_move:
                shutil.move(str(path), str(free_destination(args.target_dir, path.name)))
            print(f"\nMoved {len(to_move)} image(s) to {args.target_dir}")
        else:
            print("\nDry run: nothing was moved. Use --apply to move these images.")

    if orphan_labels:
        print("\nLabels without a matching image (not touched):")
        for path in orphan_labels:
            print(f"  - {path.name}")

    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())