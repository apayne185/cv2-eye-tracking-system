import cv2
import numpy as np


def make_accumulator(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w), dtype=np.float32)


def add_gaze_point(acc: np.ndarray, x: int, y: int) -> None:
    if 0 <= x < acc.shape[1] and 0 <= y < acc.shape[0]:
        acc[y, x] += 1.0


def render_heatmap(acc: np.ndarray, blur_kernel: int = 51) -> np.ndarray:
    """Renders a heat accumulator array into a JET colormap image."""
    heat = cv2.GaussianBlur(acc, (blur_kernel, blur_kernel), 0)
    if heat.max() > 0:
        heat = (heat / heat.max() * 255).astype(np.uint8)
    else:
        heat = np.zeros_like(heat, dtype=np.uint8)
    return cv2.applyColorMap(heat, cv2.COLORMAP_JET)


def generate_heatmap(frame: np.ndarray, gaze_points: list,
                     blur_kernel: int = 51) -> np.ndarray:
    """Builds a JET-colored heatmap from a list of (x, y) gaze points."""
    acc = make_accumulator(frame.shape[0], frame.shape[1])
    for x, y in gaze_points:
        add_gaze_point(acc, x, y)
    return render_heatmap(acc, blur_kernel)
