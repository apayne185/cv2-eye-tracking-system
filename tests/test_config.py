import json
import logging
from pathlib import Path

import pytest

from eyetrack.config import Config, ConfigError, load_config
from eyetrack.logs import JsonFormatter

EXAMPLE = Path(__file__).parent.parent / "eyetrack.example.toml"


def test_defaults():
    cfg = load_config()
    assert cfg.output_dir == Path("data")
    assert cfg.display is True
    assert "Center" in cfg.aois


def test_example_file_loads():
    cfg = load_config(EXAMPLE)
    assert cfg.aois["Center"] == (320, 100, 600, 400)


def test_file_then_overrides(tmp_path):
    path = tmp_path / "site.toml"
    path.write_text('source = "rtsp://cam/1"\nfixation_velocity = 40\n[aois]\nScreen = [0, 0, 640, 480]\n')
    cfg = load_config(path, {"source": "clip.mp4", "output_dir": None})
    assert cfg.source == "clip.mp4"            # CLI wins
    assert cfg.fixation_velocity == 40         # file beats default
    assert cfg.output_dir == Path("data")      # None override ignored
    assert cfg.aois == {"Screen": (0, 0, 640, 480)}


@pytest.mark.parametrize("toml, message", [
    ("bogus = 1", "unknown config key"),
    ('log_format = "xml"', "log_format"),
    ("fixation_velocity = -1", "must be positive"),
    ("[aois]\nA = [10, 10, 5, 20]", "x2<=x1"),
    ("[aois]\nA = [1, 2, 3]", "x1, y1, x2, y2"),
    ("source = ", "invalid TOML"),
])
def test_invalid_config_rejected(tmp_path, toml, message):
    path = tmp_path / "bad.toml"
    path.write_text(toml)
    with pytest.raises(ConfigError, match=message):
        load_config(path)


def test_missing_file_rejected(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_numeric_source_is_string():
    assert Config().update({"source": 1}).source == "1"


def test_json_log_formatter_includes_extra_fields():
    record = logging.makeLogRecord({"msg": "saved %d", "args": (3,), "levelname": "INFO",
                                    "name": "eyetrack", "frames": 3})
    entry = json.loads(JsonFormatter().format(record))
    assert entry["msg"] == "saved 3"
    assert entry["frames"] == 3
    assert entry["level"] == "INFO"
