# Images annotation visualization

A data annotation project: bounding-box labels for **drones** in photos, together with the labeling rules and small utility scripts that keep the annotations clean and consistent.

**Subject:** drones (multirotors, fixed-wing, FPV and other unmanned aerial vehicles)

**Annotation format:** YOLO (`class x_center y_center width height`, normalized to `[0, 1]`)

This repository contains annotations only. It does not include any model training code or a train/val/test split.

---

## Project structure

```
.
├── images/                   # source photos (.jpg, .jpeg, .png, .bmp, .webp)
├── labels/                   # one YOLO .txt file per image (same base name)
├── class_names.txt           # class names, one per line (line number = class ID)
├── labeling_rules.md         # rules every annotator must follow
├── shared.py                 # helpers shared by the scripts
├── check_dataset.py          # dataset check (errors + warnings + statistics)
├── clip_boxes.py             # clips boxes that stick out of the image
├── remove_duplicates.py      # removes duplicate boxes
├── move_unlabeled.py         # moves images without a label file out of images/
├── preview_annotations.py    # creates copies with colored annotation boxes
├── preview_images/           # generated visual-review images (created on demand)
└── requirements.txt
```

## Classes

| ID | Class   | Description |
|----|---------|-------------|
| 0  | `drone` | An unmanned aerial vehicle (in flight or on the ground) |

Birds, planes, helicopters, kites, balloons, drones on screens/posters/banners and reflections are **intentionally not labeled**. See [`labeling_rules.md`](labeling_rules.md) for the full rules (occlusion, motion blur, image edges, small and crowded objects, scene-specific cases).

## Label format

Each `labels/<image_name>.txt` contains one line per object:

```
<class_id> <x_center> <y_center> <width> <height>
```

- All four values are normalized to `[0, 1]` by the image width/height.
- The box must lie fully inside the image.
- An empty file means "image with no labeled objects". Every image must have a label file, even if it is empty.

Example:

```
0 0.512000 0.334000 0.041000 0.032000
0 0.187000 0.702000 0.017000 0.014000
```

## Requirements

- Python 3.8+
- [Pillow](https://pypi.org/project/Pillow/)

```bash
pip install -r requirements.txt
```

## Scripts

By default every script expects `images/`, `labels/` and `class_names.txt` next to it. Other locations can be passed with `--images`, `--labels` and `--classes` (run a script with `--help` to see its options).

### `check_dataset.py`: check the dataset

```bash
python check_dataset.py            # exit code 1 if there are errors
python check_dataset.py --strict   # exit code 1 if there are warnings too
```

Prints the number of images and boxes per class, then:

**Errors** (must be fixed):
- image cannot be opened
- missing label file
- line does not have 5 values, or contains invalid numbers
- unknown class ID (checked against `class_names.txt`)
- coordinates outside `[0, 1]`, zero or negative box size
- box extends beyond the image
- two images with the same base name (they would share one label file)

**Warnings** (should be reviewed):
- duplicate boxes of the same class (IoU > 0.95)
- box smaller than 8 px on a side (labeling rules, section 7)
- drone box larger than 25% of the image on a side (heuristic)
- label file without a matching image
- unsupported files in `images/`

Coordinates are compared with a tolerance of `1e-6` to avoid false errors caused by floating-point rounding.

### `clip_boxes.py`: clip boxes to the image

```bash
python clip_boxes.py --dry-run   # show what would change
python clip_boxes.py
```

- Boxes that stick out by a small amount (up to 2% of the image size, `--max-overshoot`) are clipped to the image boundary.
- Boxes completely outside the image are removed.
- Larger overshoots are **not** changed. They are listed for manual review, because they are usually annotation mistakes.
- All other lines are kept exactly as they are; files are written atomically.

### `remove_duplicates.py`: remove duplicate boxes

```bash
python remove_duplicates.py --dry-run
python remove_duplicates.py
```

For boxes of the same class with IoU above 0.95 (`--iou-threshold`) the first one is kept and the others are removed.

### `move_unlabeled.py`: move unannotated images away

```bash
python move_unlabeled.py           # dry run
python move_unlabeled.py --apply   # move the images
```

Images that have no label file are moved to `unlabeled_images/` (not deleted, so the step can be undone; change the folder with `--target-dir`). Labels without an image are only reported. The script refuses to run if no image has a label file (usually a wrong `--labels` path).

> An image with an **empty** label file is kept. Make sure your annotation tool exports an empty `.txt` file for images without objects, otherwise they will be treated as unannotated.

### `preview_annotations.py`: create annotated preview images

```bash
python preview_annotations.py
```

The script creates `preview_images/` and saves a copy of every image that has a matching file in `labels/`. It draws each YOLO bounding box in a distinct color for its class and adds the class name. Images without a label file are skipped; the originals and label files are never changed.

To use different folders:

```bash
python preview_annotations.py --images path/to/images --labels path/to/labels --out path/to/previews
```

## Recommended workflow

1. Annotate images following [`labeling_rules.md`](labeling_rules.md).
2. Export labels in YOLO format into `labels/`.
3. `python clip_boxes.py`: clip boxes that go slightly outside the image.
4. `python remove_duplicates.py`: remove duplicate boxes.
5. `python check_dataset.py`: fix everything that is reported.
6. `python preview_annotations.py`: inspect the colored preview images in `preview_images/`.
7. (Optional) `python move_unlabeled.py --apply`: move images that were never annotated.
8. Review 5–10% of the images a second time (section 10 of the rules).

## Quality control

Consistency matters more than perfection. The rules describe self-checks, a second-annotator comparison (target IoU ≥ 0.7 for large drones, ≥ 0.5 for small or distant ones) and how new rules are added when annotators disagree. The QC history is kept at the end of `labeling_rules.md`.

## Dataset statistics

Run `python check_dataset.py` to get the current numbers.

| Item | Value |
|------|-------|
| Images | – |
| `drone` boxes | – |