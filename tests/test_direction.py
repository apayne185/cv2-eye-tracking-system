from direction import GazeDirectionEstimator


def test_centered_iris_no_head_movement_is_origin():
    est = GazeDirectionEstimator()
    dh, dv = est.estimate(0.5, 0.5, 0.0, 0.0)
    assert abs(dh) < 1e-9
    assert abs(dv) < 1e-9


def test_left_iris_gives_negative_dir_h():
    est = GazeDirectionEstimator()
    dh, _ = est.estimate(0.2, 0.5, 0.0, 0.0)
    assert dh < 0


def test_right_iris_gives_positive_dir_h():
    est = GazeDirectionEstimator()
    dh, _ = est.estimate(0.8, 0.5, 0.0, 0.0)
    assert dh > 0


def test_head_yaw_right_adds_positive_dir_h():
    est = GazeDirectionEstimator()
    dh_no_yaw, _ = est.estimate(0.5, 0.5, 0.0, 0.0)
    dh_yaw, _    = est.estimate(0.5, 0.5, 30.0, 0.0)
    assert dh_yaw > dh_no_yaw


def test_output_clamped_to_unit_range():
    est = GazeDirectionEstimator()
    dh, dv = est.estimate(1.0, 1.0, 90.0, 90.0)
    assert -1.0 <= dh <= 1.0
    assert -1.0 <= dv <= 1.0


def test_to_screen_point_straight_ahead_is_center():
    est = GazeDirectionEstimator()
    x, y = est.to_screen_point(0.0, 0.0, 1920, 1080)
    assert x == 960
    assert y == 540


def test_to_screen_point_full_left_is_zero():
    est = GazeDirectionEstimator()
    x, _ = est.to_screen_point(-1.0, 0.0, 1920, 1080)
    assert x == 0
