"""
AWG (Arbitrary Waveform Generator) control utilities for the atommovr pipeline.

.. deprecated::
    This module has been moved to ``awg_controller.src.awg_control``.
    This shim re-exports everything for backward compatibility.
"""

from awg_controller.src.awg_control import (  # noqa: F401
    MAX_AMPLITUDE_PCT_PER_CHANNEL,
    TOTAL_DDS_CORES,
    ALL_CHANNEL_0_CORES,
    CHANNEL_0_EXCLUSIVE_CORES,
    CHANNEL_1_FULL_CORES,
    CHANNEL_1_SINGLE_CORE,
    CHANNEL_CORE_MAP,
    compute_core_assignments,
    validate_hardware_limits,
    AODSettings,
    RFRamp,
    AWGBatch,
    RFConverter,
)
