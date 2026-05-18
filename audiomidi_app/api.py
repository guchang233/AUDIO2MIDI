"""
Audio-to-MIDI 扒谱系统 - 现代化 API 接口

提供多种转录方案，从简单到复杂，满足不同场景需求

作者: AI Agent
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Callable
import numpy as np

from audiomidi_app.midi import NoteEvent


@dataclass
class TranscribeOptions:
    """转录选项"""
    # 模型选择
    use_modern_pipeline: bool = True
    use_separation: bool = False  # 是否使用源分离
    use_symbolic_decoder: bool = True
    
    # 后处理选项
    apply_postprocess: bool = True
    apply_pedal_correction: bool = True
    
    # 输出选项
    return_metadata: bool = False


class ModernTranscriber:
    """
    现代扒谱系统 - 集成最佳实践
    
    自动选择最佳可用模型，提供 production-ready 的转录能力
    """
    
    def __init__(self):
        self._initialized = False
        self._models = {}
        self._symbolic_decoder = None
        self._initialize()
    
    def _initialize(self):
        """初始化所有可用模型"""
        if self._initialized:
            return
        
        print("初始化 ModernTranscriber...")
        
        # 1. 尝试加载 PianoTranscription（最佳钢琴模型）
        try:
            from audiomidi_app.transcribe import try_piano_transcription_transcriber
            pt = try_piano_transcription_transcriber()
            if pt:
                self._models['piano'] = pt
                print("✓ PianoTranscription 已加载")
        except Exception as e:
            print(f"✗ PianoTranscription 加载失败: {e}")
        
        # 2. 尝试加载 BasicPitch（通用模型）
        try:
            from audiomidi_app.transcribe import try_basic_pitch_transcriber
            bp = try_basic_pitch_transcriber()
            if bp:
                self._models['basic_pitch'] = bp
                print("✓ BasicPitch 已加载")
        except Exception as e:
            print(f"✗ BasicPitch 加载失败: {e}")
        
        # 3. 尝试加载 Demucs（源分离）
        try:
            import demucs.pretrained
            self._models['demucs'] = True
            print("✓ Demucs 已加载")
        except Exception as e:
            print(f"✗ Demucs 加载失败: {e}")
        
        # 4. 加载 Symbolic Decoder
        try:
            from audiomidi_app.symbolic_decoder import create_symbolic_decoder
            self._symbolic_decoder = create_symbolic_decoder("default")
            print("✓ Symbolic Decoder 已加载")
        except Exception as e:
            print(f"✗ Symbolic Decoder 加载失败: {e}")
        
        self._initialized = True
    
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 44100,
        options: Optional[TranscribeOptions] = None,
    ) -> List[NoteEvent]:
        """
        转录音频到 MIDI 音符
        
        Args:
            audio: 音频样本，形状 (samples,) 或 (channels, samples)
            sample_rate: 采样率
            options: 转录选项
        
        Returns:
            NoteEvent 列表
        """
        if options is None:
            options = TranscribeOptions()
        
        # 确保音频格式正确
        audio = self._prepare_audio(audio)
        
        # 选择转录器
        transcriber = self._select_transcriber()
        
        if transcriber is None:
            raise RuntimeError("没有可用的转录模型")
        
        # 执行转录
        notes = transcriber.transcribe(audio, sample_rate)
        
        # 应用 Symbolic Decoder
        if options.use_symbolic_decoder and self._symbolic_decoder:
            notes = self._symbolic_decoder.decode(notes, tempo=120.0)
        
        # 应用后处理
        if options.apply_postprocess:
            notes = self._postprocess(notes, audio, sample_rate)
        
        # 按时间和音高排序
        notes.sort(key=lambda n: (n.start_s, n.note))
        
        return notes
    
    def _prepare_audio(self, audio: np.ndarray) -> np.ndarray:
        """准备音频格式"""
        if audio.ndim == 2:
            # 立体声转单声道
            audio = audio.mean(axis=-1)
        return audio.astype(np.float32)
    
    def _select_transcriber(self):
        """选择最佳可用转录器"""
        # 优先使用 PianoTranscription
        if 'piano' in self._models:
            return self._models['piano']
        
        # 回退到 BasicPitch
        if 'basic_pitch' in self._models:
            return self._models['basic_pitch']
        
        # 回退到 DSP
        from audiomidi_app.transcribe import HarmonicSalienceTranscriber
        return HarmonicSalienceTranscriber()
    
    def _postprocess(
        self,
        notes: List[NoteEvent],
        audio: np.ndarray,
        sample_rate: int
    ) -> List[NoteEvent]:
        """后处理"""
        try:
            from audiomidi_app.postprocess import full_postprocess
            notes = full_postprocess(notes, audio, sample_rate)
        except Exception as e:
            print(f"Postprocess failed: {e}")
        return notes
    
    def get_available_models(self) -> dict:
        """获取所有可用模型"""
        return {
            'piano': 'piano' in self._models,
            'basic_pitch': 'basic_pitch' in self._models,
            'demucs': 'demucs' in self._models,
            'symbolic_decoder': self._symbolic_decoder is not None,
        }


class SimpleTranscriber:
    """
    简单扒谱 - 直接使用原始模型，最少干预
    适合快速测试和调试
    """
    
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 44100,
    ) -> List[NoteEvent]:
        """
        直接转录，不做任何后处理
        """
        # 准备音频
        if audio.ndim == 2:
            audio = audio.mean(axis=-1)
        audio = audio.astype(np.float32)
        
        # 尝试使用 PianoTranscription
        try:
            from audiomidi_app.transcribe import try_piano_transcription_transcriber
            transcriber = try_piano_transcription_transcriber()
            if transcriber:
                return transcriber.transcribe(audio, sample_rate)
        except Exception:
            pass
        
        # 回退到 BasicPitch
        try:
            from audiomidi_app.transcribe import try_basic_pitch_transcriber
            transcriber = try_basic_pitch_transcriber()
            if transcriber:
                return transcriber.transcribe(audio, sample_rate)
        except Exception:
            pass
        
        # 最后回退到 DSP
        from audiomidi_app.transcribe import HarmonicSalienceTranscriber
        return HarmonicSalienceTranscriber().transcribe(audio, sample_rate)


def transcribe_audio(
    audio: np.ndarray,
    sample_rate: int = 44100,
    mode: str = "auto"
) -> List[NoteEvent]:
    """
    快速转录函数
    
    Args:
        audio: 音频数据
        sample_rate: 采样率
        mode: 
            - "auto": 自动选择最佳模型
            - "simple": 最小后处理
            - "full": 完整现代 pipeline
    
    Returns:
        MIDI 音符列表
    """
    if mode == "simple":
        return SimpleTranscriber().transcribe(audio, sample_rate)
    elif mode == "full":
        transcriber = ModernTranscriber()
        return transcriber.transcribe(
            audio, 
            sample_rate,
            TranscribeOptions(use_symbolic_decoder=True, apply_postprocess=True)
        )
    else:  # auto
        transcriber = ModernTranscriber()
        return transcriber.transcribe(audio, sample_rate)


# ======== 便捷函数 ========

def transcribe_with_piano(
    audio: np.ndarray,
    sample_rate: int = 44100
) -> List[NoteEvent]:
    """使用 PianoTranscription 转录（最佳钢琴效果）"""
    if audio.ndim == 2:
        audio = audio.mean(axis=-1)
    audio = audio.astype(np.float32)
    
    try:
        from audiomidi_app.transcribe import try_piano_transcription_transcriber
        transcriber = try_piano_transcription_transcriber()
        if transcriber:
            return transcriber.transcribe(audio, sample_rate)
    except Exception as e:
        print(f"PianoTranscription failed: {e}")
    
    raise RuntimeError("PianoTranscription 不可用")


def transcribe_with_basic_pitch(
    audio: np.ndarray,
    sample_rate: int = 44100
) -> List[NoteEvent]:
    """使用 BasicPitch 转录（通用模型）"""
    if audio.ndim == 2:
        audio = audio.mean(axis=-1)
    audio = audio.astype(np.float32)
    
    try:
        from audiomidi_app.transcribe import try_basic_pitch_transcriber
        transcriber = try_basic_pitch_transcriber()
        if transcriber:
            return transcriber.transcribe(audio, sample_rate)
    except Exception as e:
        print(f"BasicPitch failed: {e}")
    
    raise RuntimeError("BasicPitch 不可用")


def transcribe_with_demucs(
    audio: np.ndarray,
    sample_rate: int = 44100
) -> List[NoteEvent]:
    """使用 Demucs 源分离 + 专门模型转录（最佳效果）"""
    try:
        import demucs.pretrained
        from demucs.separate import separate_sources
    except ImportError:
        raise RuntimeError("Demucs 未安装。请运行: pip install demucs")
    
    if audio.ndim == 1:
        audio = np.stack([audio, audio])
    
    print("正在分离音频源...")
    sources = separate_sources(
        demucs.pretrained.get_model("htdemucs_ft"),
        audio,
        shifts=1,
        overlap=0.25,
    )
    sources = sources[0]
    
    print("分离完成:")
    print(f"  - Drums: {'✓' if sources.shape[0] > 1 else '✗'}")
    print(f"  - Bass: {'✓' if sources.shape[0] > 2 else '✗'}")
    print(f"  - Other: {'✓' if sources.shape[0] > 0 else '✗'}")
    print(f"  - Vocals: {'✓' if sources.shape[0] > 3 else '✗'}")
    
    # 使用主要音轨转录
    other = sources[0].numpy() if hasattr(sources[0], 'numpy') else sources[0]
    
    return transcribe_with_piano(other, sample_rate)


# ======== 使用示例 ========

"""
# 示例 1: 最简单的方式
from audiomidi_app.api import transcribe_audio
notes = transcribe_audio(audio, sample_rate)

# 示例 2: 使用特定模型
from audiomidi_app.api import (
    transcribe_with_piano,
    transcribe_with_basic_pitch,
    transcribe_with_demucs,
)

# 钢琴音频
notes = transcribe_with_piano(audio, sample_rate)

# 通用音频
notes = transcribe_with_basic_pitch(audio, sample_rate)

# 混音（需要 Demucs）
notes = transcribe_with_demucs(audio, sample_rate)

# 示例 3: 完整控制
from audiomidi_app.api import ModernTranscriber, TranscribeOptions

transcriber = ModernTranscriber()
print("可用模型:", transcriber.get_available_models())

notes = transcriber.transcribe(
    audio,
    sample_rate,
    TranscribeOptions(
        use_symbolic_decoder=True,
        apply_postprocess=True,
        apply_pedal_correction=True,
    )
)

# 示例 4: 最小干预（调试用）
from audiomidi_app.api import SimpleTranscriber
notes = SimpleTranscriber().transcribe(audio, sample_rate)
"""
