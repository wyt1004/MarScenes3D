# Annotation Formats

All text files are whitespace separated. Dimensions and centers use the LiDAR
coordinate system and metric units unless stated otherwise. Point clouds use
`X` right, `Y` forward, and `Z` up, with rows `[x, y, z, intensity]`.

## 3D detection

Each object contains 15 fields:

```text
supercategory class_name track_id attr0 attr1 attr2 attr3 dx dy dz x y z yaw point_count
```

The four `attr` fields are currently `None`. The OpenPCDet box order derived
from a row is `(x, y, z, dx, dy, dz, yaw)`. Example:

```text
vessel medium_vessel 1 None None None None 41.974 7.695 17.068 -63.677 47.859 7.206 1.883 5105
```

3D detection predictions append a confidence score to the same row:

```text
supercategory class_name track_id attr0 attr1 attr2 attr3 dx dy dz x y z yaw point_count confidence
```

## 3D tracking

Tracking annotations use a compact 12-field representation without the four
optional detection attributes:

```text
frame_id supercategory class_name track_id dx dy dz x y z yaw point_count
```

`track_id` is consistent for the same physical object within a sequence.
The reader also accepts an expanded 16-field row containing `attr0` through
`attr3` between `track_id` and `dx`.

Tracker predictions use the same compact or expanded fields. They may append a
confidence score as field 13 (compact) or field 17 (expanded); ground-truth
point counts are ignored by the evaluator and predictions without a score use
confidence `1.0`.

## 2D detection

YOLO annotations contain five fields:

```text
class_id center_x center_y width height
```

All coordinates are normalized by image width or height and must lie in
`[0, 1]`. The supplied sample uses class `0` for `vessel`.

## 3D detection evaluation

The repository evaluator uses 3D IoU `0.5` for every detection class. The
reported AP is the 40-point interpolated AP (`AP_R40`).
