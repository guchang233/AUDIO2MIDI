"""
Audio-to-MIDI API - 简洁的扒谱接口

导出所有便捷函数
"""

from audiomidi_app.api import (
    # 主要接口
    transcribe_audio,
    ModernTranscriber,
    SimpleTranscriber,
    TranscribeOptions,
    
    # 特定模型接口
    transcribe_with_piano,
    transcribe_with_basic_pitch,
    transcribe_with_demucs,
)

__all__ = [
    "transcribe_audio",
    "ModernTranscriber",
    "SimpleTranscriber",
    "TranscribeOptions",
    "transcribe_with_piano",
    "transcribe_with_basic_pitch",
    "transcribe_with_demucs",
]
