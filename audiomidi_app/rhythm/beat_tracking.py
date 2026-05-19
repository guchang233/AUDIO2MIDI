"""
Beat Tracking and Tempo Map Module
节拍跟踪和速度图 - 现代系统必须有的组件
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class BeatTrackingConfig:
    """节拍跟踪配置"""
    units_per_beat: int = 480
    beat_subdivision: int = 4
    
    enable_downbeat_detection: bool = True
    enable_tempo_estimation: bool = True
    
    min_tempo: float = 40.0
    max_tempo: float = 220.0


@dataclass(frozen=True)
class BeatInfo:
    """节拍信息"""
    time: float
    beat_number: int
    is_downbeat: bool
    confidence: float


@dataclass(frozen=True)
class TempoMap:
    """速度图"""
    bpm: float
    time_signature: tuple[int, int]
    beats: list[BeatInfo]
    
    def get_beat_at(self, time: float) -> Optional[BeatInfo]:
        """获取指定时间的节拍"""
        for beat in self.beats:
            if abs(beat.time - time) < 0.05:
                return beat
        return None
    
    def get_nearest_beat(self, time: float) -> BeatInfo:
        """获取最近的节拍"""
        if not self.beats:
            raise ValueError("No beats in tempo map")
        
        nearest = min(self.beats, key=lambda b: abs(b.time - time))
        return nearest


class BeatTracker:
    """节拍跟踪器"""
    
    def __init__(self, config: BeatTrackingConfig | None = None):
        self._cfg = config or BeatTrackingConfig()
    
    def track(self, samples: np.ndarray, sample_rate: int) -> TempoMap:
        """
        跟踪节拍
        
        Args:
            samples: 音频样本
            sample_rate: 采样率
        
        Returns:
            TempoMap: 速度图
        """
        try:
            import librosa
            
            tempo, beats = librosa.beat.beat_track(
                y=samples,
                sr=sample_rate,
                hop_length=512,
                tightness=100,
            )
            
            beat_times = librosa.frames_to_time(
                beats,
                sr=sample_rate,
                hop_length=512
            )
            
            bpm = float(tempo) if tempo > 0 else 120.0
            
            bpm = np.clip(bpm, self._cfg.min_tempo, self._cfg.max_tempo)
            
            beat_info_list = self._create_beat_info(beat_times, bpm)
            
            return TempoMap(
                bpm=bpm,
                time_signature=(4, 4),
                beats=beat_info_list,
            )
            
        except Exception as e:
            print(f"Beat tracking failed: {e}")
            return self._create_default_tempo_map()
    
    def _create_beat_info(
        self, beat_times: np.ndarray, bpm: float
    ) -> list[BeatInfo]:
        """创建节拍信息列表"""
        beats = []
        
        for i, time in enumerate(beat_times):
            is_downbeat = (i % 4) == 0
            
            beat_info = BeatInfo(
                time=float(time),
                beat_number=i,
                is_downbeat=is_downbeat,
                confidence=0.9,
            )
            beats.append(beat_info)
        
        return beats
    
    def _create_default_tempo_map(self) -> TempoMap:
        """创建默认速度图"""
        return TempoMap(
            bpm=120.0,
            time_signature=(4, 4),
            beats=[],
        )


class TransformerBeatTracker(BeatTracker):
    """基于 Transformer 的节拍跟踪器（BeatNet）
    
    注意：BeatNet API 版本可能不同，实际接口请参考：
    https://github.com/mjhydri/BeatNet
    当前实现假设 BeatNet(1, mode='online', plot=[], thread=False)
    """
    
    name = "Transformer Beat Tracker"
    
    def __init__(self, config: BeatTrackingConfig | None = None):
        self._cfg = config or BeatTrackingConfig()
        self._model = None
        self._initialize_model()
    
    def _initialize_model(self):
        try:
            from BeatNet import BeatNet
            # BeatNet API: BeatNet(1, mode='online', plot=[], thread=False)
            self._model = BeatNet(1, mode="offline", plot=[], thread=False)
        except ImportError:
            print("Warning: BeatNet not available")
            self._model = None
    
    def track(self, samples: np.ndarray, sample_rate: int) -> TempoMap:
        if self._model is None:
            return TempoMap(bpm=120.0, time_signature=(4, 4), beats=[])
        
        try:
            beat_times = self._model.process(samples)
            
            if len(beat_times) < 2:
                return TempoMap(bpm=120.0, time_signature=(4, 4), beats=[])
            
            intervals = np.diff(beat_times)
            median_interval = float(np.median(intervals))
            bpm = 60.0 / median_interval if median_interval > 0 else 120.0
            
            beat_info_list = self._create_beat_info(beat_times, bpm)
            
            return TempoMap(
                bpm=bpm,
                time_signature=(4, 4),
                beats=beat_info_list
            )
        except Exception as e:
            print(f"Transformer beat tracking failed: {e}")
            return TempoMap(bpm=120.0, time_signature=(4, 4), beats=[])


class MadmomBeatTracker(BeatTracker):
    """Madmom 节拍跟踪器 - 快速"""
    
    def __init__(self, config: BeatTrackingConfig | None = None):
        super().__init__(config)
    
    def track(self, samples: np.ndarray, sample_rate: int) -> TempoMap:
        """使用 Madmom 进行节拍跟踪"""
        try:
            import madmom
            
            # madmom 期望文件路径或 madmom.audio.Signal 对象，不接受原始 numpy array
            signal = madmom.audio.Signal(samples, sample_rate)
            
            activ_processor = madmom.features.beats.RNNBeatProcessor()
            activations = activ_processor(signal)
            
            beat_processor = madmom.features.beats.BeatTrackingProcessor(
                fps=100,
                look_ahead=0.5
            )
            beat_times = beat_processor(activations)
            
            if len(beat_times) < 2:
                return TempoMap(bpm=120.0, time_signature=(4, 4), beats=[])
            
            intervals = np.diff(beat_times)
            median_interval = float(np.median(intervals))
            bpm = 60.0 / median_interval if median_interval > 0 else 120.0
            
            beat_info_list = self._create_beat_info(beat_times, bpm)
            
            return TempoMap(
                bpm=bpm,
                time_signature=(4, 4),
                beats=beat_info_list,
            )
            
        except Exception as e:
            print(f"Madmom beat tracking failed: {e}")
            return super().track(samples, sample_rate)


class MultiModelBeatTracker:
    """多模型节拍跟踪器 - 集成多个方法"""
    
    def __init__(self, config: BeatTrackingConfig | None = None):
        self._cfg = config or BeatTrackingConfig()
        self._trackers = [
            BeatTracker(config),
            MadmomBeatTracker(config),
        ]
    
    def track(self, samples: np.ndarray, sample_rate: int) -> TempoMap:
        """多模型集成节拍跟踪"""
        all_tempo_maps = []
        
        for tracker in self._trackers:
            try:
                tempo_map = tracker.track(samples, sample_rate)
                all_tempo_maps.append(tempo_map)
            except Exception as e:
                print(f"Tracker {type(tracker).__name__} failed: {e}")
                continue
        
        if not all_tempo_maps:
            return BeatTracker(self._cfg).track(samples, sample_rate)
        
        bpm_candidates = [tm.bpm for tm in all_tempo_maps]
        bpm_median = float(np.median(bpm_candidates))
        
        all_beats = []
        for tm in all_tempo_maps:
            all_beats.extend(tm.beats)
        
        fused_beats = self._fuse_beats(all_beats)
        
        return TempoMap(
            bpm=bpm_median,
            time_signature=(4, 4),
            beats=fused_beats,
        )
    
    def _fuse_beats(self, beats: list[BeatInfo]) -> list[BeatInfo]:
        """融合多个模型的节拍"""
        if not beats:
            return []
        
        sorted_beats = sorted(beats, key=lambda b: b.time)
        
        fused = []
        current_group = [sorted_beats[0]]
        
        for i in range(1, len(sorted_beats)):
            beat = sorted_beats[i]
            
            if beat.time - current_group[-1].time < 0.05:
                current_group.append(beat)
            else:
                avg_time = np.mean([b.time for b in current_group])
                avg_confidence = np.mean([b.confidence for b in current_group])
                
                fused_beat = BeatInfo(
                    time=float(avg_time),
                    beat_number=len(fused),
                    is_downbeat=(len(fused) % 4) == 0,
                    confidence=float(avg_confidence),
                )
                fused.append(fused_beat)
                
                current_group = [beat]
        
        if current_group:
            avg_time = np.mean([b.time for b in current_group])
            avg_confidence = np.mean([b.confidence for b in current_group])
            
            fused_beat = BeatInfo(
                time=float(avg_time),
                beat_number=len(fused),
                is_downbeat=(len(fused) % 4) == 0,
                confidence=float(avg_confidence),
            )
            fused.append(fused_beat)
        
        return fused
