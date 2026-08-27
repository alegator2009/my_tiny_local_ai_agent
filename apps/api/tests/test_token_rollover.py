from __future__ import annotations

from app.config import AppConfig, ModelConfig
from app.services.indexing import estimate_token_count


def test_token_estimator_is_positive():
    assert estimate_token_count("hello world") >= 2
    assert estimate_token_count("") == 0


def test_rollover_threshold_defaults():
    cfg = AppConfig()
    assert cfg.rollover_config.pre_rollover_threshold == 0.80
    assert cfg.rollover_config.hard_rollover_threshold == 0.92
    assert cfg.rollover_config.pre_rollover_threshold < cfg.rollover_config.hard_rollover_threshold


def test_model_request_timeout_bounds():
    assert ModelConfig().request_timeout_sec == 240
    assert ModelConfig(request_timeout_sec=1).request_timeout_sec == 5
    assert ModelConfig(request_timeout_sec=120).request_timeout_sec == 120
    assert ModelConfig(request_timeout_sec=9999).request_timeout_sec == 600
