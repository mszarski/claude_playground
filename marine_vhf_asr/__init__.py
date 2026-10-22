"""
Marine VHF radio transcription.

Data generation, augmentation and evaluation tooling for fine-tuning an ASR
model (Voxtral Mini 3B) on maritime radio traffic: VTS, pilots, port control,
coastguard and ship-to-ship communications.
"""

from .metrics import corpus_wer, slot_accuracy
from .normalize import normalize_transcript
from .smcp import Dialogue, DialogueGenerator, Turn
from .vhf_channel import ChannelConfig, simulate_vhf

__all__ = [
    "normalize_transcript",
    "corpus_wer",
    "slot_accuracy",
    "Dialogue",
    "DialogueGenerator",
    "Turn",
    "ChannelConfig",
    "simulate_vhf",
]
