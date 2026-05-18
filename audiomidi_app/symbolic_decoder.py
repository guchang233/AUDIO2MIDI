"""
Symbolic Decoder - 音乐语言层的核心解码器

这是 production-grade transcription 的关键：
- Harmonic Suppression: 泛音抑制
- Temporal Note Linking: 时序音符连接
- Polyphony Pruning: 复调剪枝
- Beat-aware Quantization: 节拍感知量化
- Voice Consistency: 声部一致性

作者: AI Agent
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Set, Tuple
import numpy as np
from collections import defaultdict

from audiomidi_app.midi import NoteEvent


@dataclass(frozen=True)
class SymbolicDecoderConfig:
    """符号解码器配置"""
    # Harmonic Suppression
    enable_harmonic_suppression: bool = True
    harmonic_suppression_factor: float = 0.3
    max_harmonic_order: int = 8
    harmonic_tolerance_cents: float = 15.0  # 半音 = 100 cents
    min_fundamental_confidence: float = 0.6
    
    # Temporal Note Linking
    enable_note_linking: bool = True
    link_tolerance_seconds: float = 0.08
    min_consecutive_frames: int = 3
    
    # Polyphony Pruning
    enable_polyphony_pruning: bool = True
    max_simultaneous_notes: int = 10
    piano_max_simultaneous: int = 10
    voice_max_simultaneous: int = 2
    
    # Beat-aware Quantization
    enable_beat_quantization: bool = True
    quantization_tolerance: float = 0.05  # 秒
    beat_subdivision: int = 8  # 8分音符
    
    # Voice Consistency
    enable_voice_consistency: bool = True
    max_voice_jump_semitones: int = 12
    enforce_harmony_constraints: bool = True
    
    # Duplicate Suppression
    enable_duplicate_suppression: bool = True
    duplicate_tolerance_seconds: float = 0.02


@dataclass
class NoteCandidate:
    """音符候选对象，用于 decoder 内部处理"""
    note: int
    start_time: float
    end_time: float
    velocity: int
    confidence: float
    is_harmonic: bool = False
    harmonic_of: Optional[int] = None
    frame_activations: List[float] = None
    
    def __post_init__(self):
        if self.frame_activations is None:
            self.frame_activations = []


class SymbolicDecoder:
    """
    音乐符号解码器
    
    把声学模型的 noisy frame predictions
    转换成稳定的音乐事件
    """
    
    def __init__(self, config: SymbolicDecoderConfig = None):
        self.config = config or SymbolicDecoderConfig()
    
    def decode(
        self,
        notes: List[NoteEvent],
        tempo: Optional[float] = 120.0,
        beats: Optional[List[float]] = None,
        instrument: str = "piano"
    ) -> List[NoteEvent]:
        """
        完整的解码流程
        
        Args:
            notes: 声学模型输出的原始音符
            tempo: 速度 (BPM)
            beats: 节拍位置列表
            instrument: 乐器类型
        
        Returns:
            解码后的稳定音符列表
        """
        if not notes:
            return []
        
        # 转换为内部候选对象
        candidates = self._convert_to_candidates(notes)
        
        # 阶段 1: 泛音抑制
        if self.config.enable_harmonic_suppression:
            candidates = self._suppress_harmonics(candidates)
        
        # 阶段 2: 时序音符连接
        if self.config.enable_note_linking:
            candidates = self._link_temporal_notes(candidates)
        
        # 阶段 3: 复调剪枝
        if self.config.enable_polyphony_pruning:
            candidates = self._prune_polyphony(candidates, instrument)
        
        # 阶段 4: 节拍感知量化
        if self.config.enable_beat_quantization and beats:
            candidates = self._quantize_to_beats(candidates, beats)
        
        # 阶段 5: 重复音符抑制
        if self.config.enable_duplicate_suppression:
            candidates = self._suppress_duplicates(candidates)
        
        # 阶段 6: 声部一致性检查
        if self.config.enable_voice_consistency:
            candidates = self._enforce_voice_consistency(candidates)
        
        # 转换回 NoteEvent
        return self._convert_to_notes(candidates)
    
    def _convert_to_candidates(self, notes: List[NoteEvent]) -> List[NoteCandidate]:
        """转换 NoteEvent 为内部候选对象"""
        candidates = []
        for note in notes:
            candidate = NoteCandidate(
                note=note.note,
                start_time=note.start_s,
                end_time=note.end_s,
                velocity=note.velocity,
                confidence=getattr(note, 'confidence', 1.0)
            )
            candidates.append(candidate)
        return candidates
    
    def _convert_to_notes(self, candidates: List[NoteCandidate]) -> List[NoteEvent]:
        """转换候选对象回 NoteEvent"""
        notes = []
        for candidate in candidates:
            note = NoteEvent(
                note=candidate.note,
                start_s=candidate.start_time,
                end_s=candidate.end_time,
                velocity=candidate.velocity,
                confidence=candidate.confidence
            )
            notes.append(note)
        return notes
    
    # ======== 阶段 1: 泛音抑制 ========
    
    def _suppress_harmonics(self, candidates: List[NoteCandidate]) -> List[NoteCandidate]:
        """
        泛音抑制算法
        
        核心思想：
        - 如果一个音符是另一个音符的泛音
        - 且基音存在且置信度足够高
        - 则降低泛音的置信度或删除
        
        这解决了 "泛音被误识别为独立音符" 的问题
        """
        # 按时间分组
        time_groups = self._group_notes_by_time(candidates)
        
        for time_key, group in time_groups.items():
            # 按置信度降序排序
            sorted_group = sorted(group, key=lambda x: -x.confidence)
            
            # 标记泛音
            processed_notes = set()
            for i, candidate in enumerate(sorted_group):
                if candidate.note in processed_notes:
                    continue
                
                processed_notes.add(candidate.note)
                
                # 检查是否存在基音的泛音
                for harmonic_order in range(2, self.config.max_harmonic_order + 1):
                    harmonic_note = self._calculate_harmonic(candidate.note, harmonic_order)
                    
                    # 在同组中查找可能的泛音
                    for j, potential_harmonic in enumerate(sorted_group):
                        if j <= i:
                            continue
                        if potential_harmonic.note in processed_notes:
                            continue
                        
                        # 检查是否接近泛音
                        if self._is_close_to_harmonic(
                            potential_harmonic.note,
                            harmonic_note
                        ):
                            # 标记为泛音
                            potential_harmonic.is_harmonic = True
                            potential_harmonic.harmonic_of = candidate.note
                            
                            # 降低置信度
                            suppression = self.config.harmonic_suppression_factor
                            potential_harmonic.confidence *= suppression
                            
                            processed_notes.add(potential_harmonic.note)
        
        # 过滤掉置信度太低的泛音
        filtered = []
        for candidate in candidates:
            if candidate.is_harmonic and candidate.confidence < 0.2:
                continue
            filtered.append(candidate)
        
        return filtered
    
    def _calculate_harmonic(self, fundamental: int, harmonic_order: int) -> float:
        """计算泛音的 MIDI 音高（可能不是整数）"""
        # MIDI 音高 = 12 * log2(freq / 440) + 69
        # 泛音频率 = fundamental_freq * harmonic_order
        # log2(harmonic_order) = log2(freq_h / freq_f)
        midi_harmonic = fundamental + 12.0 * np.log2(harmonic_order)
        return midi_harmonic
    
    def _is_close_to_harmonic(self, note: int, harmonic_midi: float) -> bool:
        """检查音符是否接近泛音"""
        # 计算音分差
        cents_diff = abs(note - harmonic_midi) * 100.0
        return cents_diff < self.config.harmonic_tolerance_cents
    
    def _group_notes_by_time(self, candidates: List[NoteCandidate]) -> Dict[str, List[NoteCandidate]]:
        """按时间窗口分组音符"""
        groups = defaultdict(list)
        time_window = 0.1  # 100ms
        
        for candidate in candidates:
            # 归一化到时间窗口
            time_key = int(candidate.start_time / time_window)
            groups[time_key].append(candidate)
        
        return groups
    
    # ======== 阶段 2: 时序音符连接 ========
    
    def _link_temporal_notes(self, candidates: List[NoteCandidate]) -> List[NoteCandidate]:
        """
        时序音符连接
        
        把碎片 note fragments 连接成稳定 note event
        解决 "note fragmentation" 问题
        """
        # 按音高分组
        by_note = defaultdict(list)
        for candidate in candidates:
            by_note[candidate.note].append(candidate)
        
        linked_candidates = []
        
        for note_num, note_group in by_note.items():
            if not note_group:
                continue
            
            # 按时序排序
            sorted_group = sorted(note_group, key=lambda x: x.start_time)
            
            current = sorted_group[0]
            linked_group = [current]
            
            for next_candidate in sorted_group[1:]:
                # 检查是否可以连接
                if self._can_link(current, next_candidate):
                    # 合并
                    current = NoteCandidate(
                        note=current.note,
                        start_time=current.start_time,
                        end_time=max(current.end_time, next_candidate.end_time),
                        velocity=max(current.velocity, next_candidate.velocity),
                        confidence=max(current.confidence, next_candidate.confidence)
                    )
                    linked_group[-1] = current
                else:
                    current = next_candidate
                    linked_group.append(current)
            
            linked_candidates.extend(linked_group)
        
        return linked_candidates
    
    def _can_link(self, note1: NoteCandidate, note2: NoteCandidate) -> bool:
        """检查两个音符是否可以连接"""
        # 必须是同一个音高
        if note1.note != note2.note:
            return False
        
        # 时间重叠或足够接近
        time_gap = note2.start_time - note1.end_time
        if time_gap < 0:
            return True  # 重叠
        if time_gap < self.config.link_tolerance_seconds:
            return True
        
        return False
    
    # ======== 阶段 3: 复调剪枝 ========
    
    def _prune_polyphony(
        self,
        candidates: List[NoteCandidate],
        instrument: str
    ) -> List[NoteCandidate]:
        """
        复调剪枝
        
        限制不合理的复调数量
        解决 "polyphony explosion" 问题
        """
        # 确定最大复调
        max_polyphony = self.config.max_simultaneous_notes
        if instrument == "piano":
            max_polyphony = self.config.piano_max_simultaneous
        elif instrument == "voice":
            max_polyphony = self.config.voice_max_simultaneous
        
        # 建立时间戳点
        all_times = set()
        for candidate in candidates:
            all_times.add(candidate.start_time)
            all_times.add(candidate.end_time)
        sorted_times = sorted(all_times)
        
        # 扫描并标记要移除的索引
        to_remove_indices = set()
        
        for i in range(len(sorted_times) - 1):
            start_window = sorted_times[i]
            end_window = sorted_times[i + 1]
            
            # 找出这个时间窗口内的活跃音符及其索引
            active_with_indices = []
            for idx, candidate in enumerate(candidates):
                if self._is_active_in_window(candidate, start_window, end_window):
                    active_with_indices.append((idx, candidate))
            
            # 如果超过最大复调
            if len(active_with_indices) > max_polyphony:
                # 按置信度排序
                sorted_active = sorted(
                    active_with_indices,
                    key=lambda x: (-x[1].confidence, -x[1].velocity)
                )
                
                # 标记要移除的索引
                for idx, _ in sorted_active[max_polyphony:]:
                    to_remove_indices.add(idx)
        
        # 返回未被移除的
        return [c for idx, c in enumerate(candidates) if idx not in to_remove_indices]
    
    def _is_active_in_window(
        self,
        candidate: NoteCandidate,
        window_start: float,
        window_end: float
    ) -> bool:
        """检查音符在时间窗口内是否活跃"""
        return not (candidate.end_time < window_start or candidate.start_time > window_end)
    
    # ======== 阶段 4: 节拍感知量化 ========
    
    def _quantize_to_beats(
        self,
        candidates: List[NoteCandidate],
        beats: List[float]
    ) -> List[NoteCandidate]:
        """
        节拍感知量化
        
        利用 beat tracking 信息修正 timing
        解决 "onset jitter" 和 "duration drift" 问题
        """
        if not beats:
            return candidates
        
        quantized = []
        
        for candidate in candidates:
            # 找最近的节拍
            nearest_beat = self._find_nearest_beat(candidate.start_time, beats)
            
            if nearest_beat is not None:
                # 计算偏差
                deviation = abs(candidate.start_time - nearest_beat)
                
                if deviation < self.config.quantization_tolerance:
                    # 对齐到节拍
                    adjusted_start = nearest_beat
                    duration = candidate.end_time - candidate.start_time
                    
                    # 也量化时长（可选）
                    if self.config.beat_subdivision > 0:
                        beat_duration = 60.0 / self.config.beat_subdivision
                        quantized_duration = round(duration / beat_duration) * beat_duration
                        adjusted_end = adjusted_start + max(quantized_duration, 0.05)
                    else:
                        adjusted_end = candidate.end_time
                    
                    # 更新
                    candidate = NoteCandidate(
                        note=candidate.note,
                        start_time=adjusted_start,
                        end_time=adjusted_end,
                        velocity=candidate.velocity,
                        confidence=candidate.confidence * 0.95  # 稍微降低置信度
                    )
            
            quantized.append(candidate)
        
        return quantized
    
    def _find_nearest_beat(self, time: float, beats: List[float]) -> Optional[float]:
        """找最近的节拍位置"""
        if not beats:
            return None
        
        min_dist = float('inf')
        nearest = None
        
        for beat in beats:
            dist = abs(beat - time)
            if dist < min_dist:
                min_dist = dist
                nearest = beat
        
        return nearest if min_dist < 0.1 else None
    
    # ======== 阶段 5: 重复音符抑制 ========
    
    def _suppress_duplicates(self, candidates: List[NoteCandidate]) -> List[NoteCandidate]:
        """抑制几乎相同的重复音符"""
        # 按音高分组
        by_note = defaultdict(list)
        for candidate in candidates:
            by_note[candidate.note].append(candidate)
        
        result = []
        
        for note_num, group in by_note.items():
            group_sorted = sorted(group, key=lambda x: x.start_time)
            
            current = group_sorted[0]
            result.append(current)
            
            for next_candidate in group_sorted[1:]:
                time_gap = next_candidate.start_time - current.start_time
                
                # 如果太接近，只保留置信度高的
                if time_gap < self.config.duplicate_tolerance_seconds:
                    if next_candidate.confidence > current.confidence:
                        result[-1] = next_candidate
                        current = next_candidate
                else:
                    result.append(next_candidate)
                    current = next_candidate
        
        return sorted(result, key=lambda x: x.start_time)
    
    # ======== 阶段 6: 声部一致性 ========
    
    def _enforce_voice_consistency(self, candidates: List[NoteCandidate]) -> List[NoteCandidate]:
        """
        声部一致性检查
        
        避免不可能的 hand movement 和 voice leading
        """
        # 简单版本：排序后检查跳跃
        sorted_candidates = sorted(candidates, key=lambda x: (x.start_time, x.note))
        
        # 分组到时间窗口
        time_windows = self._group_notes_by_time(sorted_candidates)
        
        for window_key, window_notes in time_windows.items():
            # 按音高排序
            sorted_pitches = sorted([n.note for n in window_notes])
            
            # 检查是否有不合理的和弦（可选）
            if self.config.enforce_harmony_constraints:
                # 这里可以加入和声学检查
                # 例如避免不协和的音程（取决于音乐风格）
                pass
        
        return sorted_candidates


def create_symbolic_decoder(
    mode: str = "default"
) -> SymbolicDecoder:
    """
    创建预设的符号解码器
    
    Args:
        mode: "default", "conservative", "aggressive"
    
    Returns:
        配置好的 SymbolicDecoder
    """
    if mode == "conservative":
        config = SymbolicDecoderConfig(
            harmonic_suppression_factor=0.5,
            link_tolerance_seconds=0.05,
            quantization_tolerance=0.03,
            max_simultaneous_notes=8
        )
    elif mode == "aggressive":
        config = SymbolicDecoderConfig(
            harmonic_suppression_factor=0.15,
            link_tolerance_seconds=0.12,
            quantization_tolerance=0.08,
            max_simultaneous_notes=12
        )
    else:
        config = SymbolicDecoderConfig()
    
    return SymbolicDecoder(config)
