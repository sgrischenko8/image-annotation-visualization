# Labeling Rules: Drone Detection in Photos

**Version:** 1.0
**Subject:** drones (multirotors, fixed-wing, FPV and other unmanned aerial vehicles)
**Task:** object detection (bounding boxes)

---

## 1. Goal

The dataset is used to train an object detection model that finds **drones** in photos. Consistent labeling matters more than perfect labeling: the same situation must always be labeled the same way, by every annotator.

## 2. Classes

| ID | Class | What it is |
|----|-------|-----------|
| 0 | `drone` | An unmanned aerial vehicle: quadcopters and other multirotors, fixed-wing drones, FPV drones, VTOL drones; in flight or standing on the ground |

### What is NOT labeled

- Birds, bats, insects
- Airplanes, helicopters with a pilot, paragliders, hot-air balloons, kites, balloons
- Drones in advertising, logos, banners, illustrations, toys' packaging
- Drones shown on screens, posters or monitors
- Reflections of drones in windows, water or glass
- Remote controllers, goggles, operators and other equipment (only the drone itself is labeled)
- Objects that look like a drone but are not one (antennas, lamps, dark spots on the sky, aircraft lights)

> Rule of thumb: if you cannot say with reasonable confidence that it is a drone, do not label it.

Unlabeled objects are **intentional**: the model must learn that "a small dark object in the sky" is not automatically a `drone`. Because of this, the class definition above must be applied strictly and identically across all images.

## 3. How to draw a bounding box

1. The box is **axis-aligned** (a rectangle with horizontal and vertical sides).
2. It must be **tight**: the edges touch the outermost visible pixels of the object. No large gaps, no cutting into the object.
3. Include everything that belongs to the drone:
   - body, arms, **propellers** (including blurred spinning propellers), landing gear, camera/gimbal, antennas that are part of the drone, wings and tail for fixed-wing models
   - a payload that is physically attached to the drone (camera, package, sprayer tank)
   - Do not include the shadow, the reflection, contrails or a light glow around the drone.
4. A drone held in someone's hand or standing on the ground is still labeled; the box covers only the drone, not the hand, the person or the launch pad.
5. Boxes of different objects may overlap. Do not shrink a box to avoid overlap.
6. One object = one box. Never draw one box around a group of drones.
7. Start the box from the top-left visible extremity and finish at the bottom-right, then check that no part of the drone is outside the box.

## 4. Partially visible objects (occlusion)

- If **at least ~30%** of the drone is visible, label it. Draw the box around the **visible part only** (do not extrapolate hidden parts).
- If **less than ~30%** is visible, or you cannot tell what it is, do not label it.
- A drone occluded by another drone: label both, each with the box around its own visible part.
- A drone partly hidden behind a tree, wire, building or window frame: label the visible part if it is clearly a drone.
- A drone fully covered by another object is not labeled.

## 5. Blurred and low-quality drones (motion blur, out of focus)

Drones are often small, fast and blurred.

- **Drone is blurred but you can clearly tell it is a drone** (typical shape, propellers, correct context): label it. Draw the box around the whole blurred shape, including the smear, as long as the smear looks like part of the drone's own shape.
- **Drone is stretched by motion blur:** box around the full visible streak, not only the sharpest point.
- **You are unsure it is a drone** (could be a bird, an insect, a dust spot, a sensor stain, a distant aircraft): do **not** guess. Leave it unlabeled.
- **Do not use the context to invent a drone.** If the drone is not visible, it is not labeled, even if people on the ground are clearly looking up or holding a controller.
- Use the same decision for all scenes: "Can an independent annotator, looking only at this crop, say it is a drone with reasonable confidence?" If yes, label it. If no, leave it.

## 6. Objects at the image edge

- If a drone is **cut off by the image border**, label the **visible part**. The box goes up to the image edge (coordinates are clipped to the image boundaries).
- Do not make the box extend beyond the image.
- Apply the visibility threshold from section 4: if only a tiny piece (one propeller, a part of a wing) is visible at the edge, do not label it.
- A drone at the edge that is clearly visible as a drone: label it normally, even if only half of it is in the frame.

## 7. Small, distant and crowded objects

- **Small objects:** label them if they are at least ~8×8 pixels and clearly recognizable as drones. Smaller than this: leave unlabeled.
- **Distant drones** in wide shots: label them if you can tell that they are drones and not birds. If you cannot, leave them unlabeled.
- **Swarms and light shows:** label each separately visible drone. Do not draw a group box. If drones are too small or too dense to be separated into individuals, leave those unlabeled rather than drawing a box around the group.
- **Multiple drones** in the frame: label every drone that meets the rules above, in flight or on the ground.

## 8. Uncertain cases

Only the class `drone` is used, so there is no separate class for uncertain objects. Make a decision:

- Clearly a drone (see section 5, last rule): label it.
- Not sure: leave it unlabeled.
- Never draw a box "just in case", and never leave clear drones unlabeled because the work is tedious.

Tip: if you notice many recurring uncertain cases in a batch, discuss them and add the decision as a new rule to this document (see section 10).

## 9. Scene-specific notes

| Scene | What to watch for | Special cases |
|-------|-------------------|---------------|
| Open sky | Drone is usually small and far away | Easy to confuse with birds: look for straight edges, propellers, steady position |
| City / buildings background | Drone can blend with windows, cables and antennas | Label the visible part if occluded by a cable or a pole |
| Night / lights | Often only LEDs are visible | Label only if the drone body or clear drone shape can be seen; a lone light is not enough |
| Ground / launch | Drone on the ground or in a hand | Label the drone, not the hand, the pad or the case |
| Show / swarm | Many drones in a formation | Label each separately visible drone; no group boxes |

## 10. Quality control

- **Self-check:** before submitting, look at each image once more and check: no missing drones, boxes are tight, no bird or other object is labeled as a drone.
- **Consistency check:** review 5–10% of the images a second time after a break; fix systematic differences (e.g. boxes that are too loose or that skip propellers).
- **Second annotator (optional):** for a sample of images, compare two independent annotations. Compute IoU between matching boxes; a target of **IoU ≥ 0.5** for small or distant drones and **≥ 0.7** for large, clear drones is reasonable (small boxes change IoU a lot with a few pixels).
- **Disagreements:** if two annotators disagree on a case, discuss it and **add the decision as a new rule** to this document.
- **Automated checks:** run `python clip_boxes.py`, `python remove_duplicates.py` and `python check_dataset.py` after annotating (see the README). Fix every error reported by `check_dataset.py`.
- **Version control:** every rule change increases the document version and is noted in the changelog below.

## 11. Common mistakes to avoid

- Labeling birds, insects, planes or kites as `drone`.
- Loose boxes with large background margins, or boxes that cut off propellers or landing gear.
- Labeling a drone that is not actually visible (guessing from context).
- Drawing one box around several drones.
- Extending the box to hidden parts.
- Forgetting the drone because it is small.
- Including the shadow, a light glow or the operator's hand in the box.
- Inconsistent treatment of the same situation across images.

## 12. Output format

- Format: YOLO (`class x_center y_center width height`, normalized to `[0, 1]`). This is the only supported format.
- Classes (see `class_names.txt`): `0 drone`. This is the only class.
- One annotation file per image, same base file name as the image.
- Empty file allowed: image with no labeled objects.

## Changelog

| Version | Change |
|---------|--------|
| 1.0 | First version of the rules |

## QC History

| Date | Action | Result |
|------|--------|--------|
| – | – | No entries yet |