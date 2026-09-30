"""Shared Ultralytics input and class-selection helpers."""
import numpy as np


ALL_TARGETS = {"", "*", "all", "any"}


def model_names(model):
    names = model.names
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    return {index: str(value) for index, value in enumerate(names)}


def class_ids(names, target):
    if target is None:
        return None
    values = [target] if isinstance(target, (int, str)) else target
    tokens = [token.strip() for value in values
              for token in str(value).split(",") if token.strip()]
    if not tokens or len(tokens) == 1 and tokens[0].lower() in ALL_TARGETS:
        return None
    by_name = {name.casefold(): index for index, name in names.items()}
    selected, unknown = [], []
    for token in tokens:
        if token.isdigit() and int(token) in names:
            selected.append(int(token))
        elif token.casefold() in by_name:
            selected.append(by_name[token.casefold()])
        else:
            unknown.append(token)
    if unknown:
        available = ", ".join(f"{key}:{value}" for key, value in names.items())
        raise ValueError(
            f"YOLO unknown target {unknown}; available classes: {available}"
        )
    return sorted(set(selected))


def bgr_image(color):
    color = np.asarray(color)
    if color.ndim != 3 or color.shape[2] != 3:
        raise ValueError("YOLO requires an HxWx3 RGB image")
    return np.ascontiguousarray(color[..., ::-1])
