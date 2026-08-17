"""RCR-YOLO: Reconstruction-assisted Context Reasoning YOLO.

Lightweight hard-object (small / occluded / blurred) detection for
unstructured indoor service-robot scenes.

Modules:
    ORBIn  - training-time-only object reconstruction branch (zero inference cost)
    LCR    - lightweight easy-help-hard context reasoning
    LCRBase- degraded fallback (confidence/graph-free scene context augmentation)
    MRFE   - multiple receptive field adaptive feature enhancement (neck plug-in)
    GSConv - ghost shuffle convolution (lightweight building block)
"""

from .common import GSConv
from .orb_in import ORBIn
from .lcr import LCR, LCRBase
from .mrfe import MRFE

__all__ = ["GSConv", "ORBIn", "LCR", "LCRBase", "MRFE"]
