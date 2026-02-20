"""
DDS execution strategies for the Spectrum Instrumentation AWG card.

.. deprecated::
    This module has been moved to ``awg_controller.src.dds_strategies``.
    This shim re-exports everything for backward compatibility.
"""

from awg_controller.src.dds_strategies import (  # noqa: F401
    RampConfig,
    PatternConfig,
    CameraTriggerConfig,
    MAX_SAFE_TRIGGER_LEVEL_V,
    _core_const,
    _group_ramps_by_channel,
    DDSStrategy,
    DDSStreamingStrategy,
    DDSRampStrategy,
    DDSPatternStrategy,
    DDSCameraTriggeredStrategy,
    STRATEGY_REGISTRY,
    get_strategy,
)
