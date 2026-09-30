"""Shared geometry operations for grasp pipelines and application workflows."""


def expand_boxes(boxes, image_shape, scale=1.25):
    """Scale xyxy boxes about their centres and clip them to the image."""
    height, width = image_shape[:2]
    expanded = []
    for x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        box_width, box_height = (x2 - x1) * scale, (y2 - y1) * scale
        expanded.append([
            max(0, int(cx - box_width / 2)),
            max(0, int(cy - box_height / 2)),
            min(width, int(cx + box_width / 2)),
            min(height, int(cy + box_height / 2)),
        ])
    return expanded
