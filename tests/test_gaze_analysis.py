import numpy as np
from gaze_analysis import make_accumulator, add_gaze_point, render_heatmap


def test_accumulator_shape_and_dtype():
    acc = make_accumulator(480, 640)
    assert acc.shape == (480, 640)
    assert acc.dtype == np.float32


def test_add_gaze_point_increments():
    acc = make_accumulator(100, 100)
    add_gaze_point(acc, 50, 50)
    assert acc[50, 50] == 1.0
    add_gaze_point(acc, 50, 50)
    assert acc[50, 50] == 2.0


def test_add_gaze_point_out_of_bounds_ignored():
    acc = make_accumulator(100, 100)
    add_gaze_point(acc, 200, 200)
    add_gaze_point(acc, -1, 50)
    assert acc.max() == 0.0


def test_render_heatmap_output_shape():
    acc = make_accumulator(480, 640)
    add_gaze_point(acc, 320, 240)
    heatmap = render_heatmap(acc)
    assert heatmap.shape == (480, 640, 3)


def test_render_empty_accumulator_does_not_raise():
    acc = make_accumulator(100, 100)
    heatmap = render_heatmap(acc)
    assert heatmap.shape == (100, 100, 3)
