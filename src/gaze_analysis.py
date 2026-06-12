import cv2
import numpy as np


def generate_heatmap(frame, gaze_points, blur_kernel=51):
    """
    Builds a JET-colored heatmap from accumulated gaze points.
    blur_kernel controls spatial smoothing (must be odd).
    """
    heat = np.zeros(frame.shape[:2], dtype=np.float32)
    for x, y in gaze_points:
        if 0 <= x < frame.shape[1] and 0 <= y < frame.shape[0]:
            heat[y, x] += 1.0

    heat = cv2.GaussianBlur(heat, (blur_kernel, blur_kernel), 0)
    if heat.max() > 0:
        heat = (heat / heat.max() * 255).astype(np.uint8)
    else:
        heat = np.zeros_like(heat, dtype=np.uint8)

    return cv2.applyColorMap(heat, cv2.COLORMAP_JET)
