"""
Per-deployment settings.

Precedence: built-in defaults < TOML config file < command-line flags.
See eyetrack.example.toml for every key.
"""

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

from .aoi import DEFAULT_AOIS
from .calibration import DEFAULT_CALIB_PATH
from .eye_tracker import EAR_BLINK_THRESHOLD, FIXATION_VEL_PX_PER_SEC, MIN_FIXATION_SECS
from .gaze_classifier import DEFAULT_MODEL_PATH


class ConfigError(ValueError):
    pass


@dataclass
class Config:
    source: str = "0"
    output_dir: Path = Path("data")
    display: bool = True
    export_ply: bool = False
    max_frames: int | None = None

    calibration_path: Path = DEFAULT_CALIB_PATH
    classifier_path: Path = DEFAULT_MODEL_PATH

    ear_blink_threshold: float = EAR_BLINK_THRESHOLD
    fixation_velocity: float = FIXATION_VEL_PX_PER_SEC
    min_fixation_secs: float = MIN_FIXATION_SECS

    aois: dict[str, tuple[int, int, int, int]] = field(
        default_factory=lambda: dict(DEFAULT_AOIS))

    log_level: str = "INFO"
    log_format: str = "text"   # "text" or "json"

    def update(self, values: dict) -> "Config":
        """Applies non-None values, coercing paths and validating keys."""
        known = {f.name: f for f in fields(self)}
        for key, value in values.items():
            if value is None:
                continue
            if key not in known:
                raise ConfigError(f"unknown config key {key!r}")
            if key.endswith(("_dir", "_path")):
                value = Path(value)
            elif key == "aois":
                value = _parse_aois(value)
            elif key == "source":
                value = str(value)
            setattr(self, key, value)
        self._validate()
        return self

    def _validate(self) -> None:
        if self.log_format not in ("text", "json"):
            raise ConfigError(f"log_format must be 'text' or 'json', not {self.log_format!r}")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ConfigError("max_frames must be positive")
        for name in ("ear_blink_threshold", "fixation_velocity", "min_fixation_secs"):
            if getattr(self, name) <= 0:
                raise ConfigError(f"{name} must be positive")


def _parse_aois(raw: dict) -> dict[str, tuple[int, int, int, int]]:
    aois = {}
    for name, box in raw.items():
        if len(box) != 4:
            raise ConfigError(f"AOI {name!r} must be [x1, y1, x2, y2]")
        x1, y1, x2, y2 = (int(v) for v in box)
        if x2 <= x1 or y2 <= y1:
            raise ConfigError(f"AOI {name!r} has x2<=x1 or y2<=y1")
        aois[name] = (x1, y1, x2, y2)
    return aois


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    cfg = Config()
    if path is not None:
        path = Path(path)
        try:
            data = tomllib.loads(path.read_text())
        except FileNotFoundError as e:
            raise ConfigError(f"config file not found: {path}") from e
        except tomllib.TOMLDecodeError as e:
            raise ConfigError(f"invalid TOML in {path}: {e}") from e
        cfg.update(data)
    return cfg.update(overrides or {})
