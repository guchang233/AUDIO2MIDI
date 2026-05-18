from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from audiomidi_app.midi import NoteEvent


@dataclass(frozen=True)
class VoiceSeparationConfig:
    max_voices: int = 4
    pitch_weight: float = 1.0
    time_weight: float = 0.5
    velocity_weight: float = 0.2
    chord_time_threshold: float = 0.03
    voice_merge_threshold: float = 0.3
    melody_bias: float = 0.3
    hand_smoothing: float = 0.7


@dataclass
class Voice:
    notes: list[NoteEvent] = field(default_factory=list)
    pitches: list[int] = field(default_factory=list)
    centers: list[float] = field(default_factory=list)
    
    @property
    def avg_pitch(self) -> float:
        return np.mean(self.pitches) if self.pitches else 60.0
    
    @property
    def avg_center(self) -> float:
        return np.mean(self.centers) if self.centers else 60.0
    
    def add_note(self, note: NoteEvent) -> None:
        self.notes.append(note)
        self.pitches.append(note.note)
        center = note.note
        if self.centers:
            center = 0.7 * self.centers[-1] + 0.3 * note.note
        self.centers.append(center)


def compute_voice_cost(
    note: NoteEvent,
    voice: Voice,
    config: VoiceSeparationConfig,
) -> float:
    if not voice.notes:
        return 100.0
    
    last_note = voice.notes[-1]
    
    pitch_diff = abs(note.note - last_note.note)
    time_diff = abs(note.start_s - last_note.end_s)
    velocity_diff = abs(note.velocity - last_note.velocity)
    
    cost = (
        config.pitch_weight * pitch_diff +
        config.time_weight * time_diff * 100 +
        config.velocity_weight * velocity_diff
    )
    
    return cost


def chord_grouping(events: list[NoteEvent], threshold: float = 0.03) -> list[list[int]]:
    if not events:
        return []
    
    sorted_events = sorted(enumerate(events), key=lambda x: x[1].start_s)
    
    groups: list[list[int]] = []
    current_group: list[int] = [sorted_events[0][0]]
    current_time = sorted_events[0][1].start_s
    
    for idx, note in sorted_events[1:]:
        if note.start_s - current_time <= threshold:
            current_group.append(idx)
        else:
            if len(current_group) > 1:
                groups.append(current_group)
            current_group = [idx]
            current_time = note.start_s
    
    if len(current_group) > 1:
        groups.append(current_group)
    
    return groups


def assign_voices_dp(
    events: list[NoteEvent],
    config: VoiceSeparationConfig | None = None,
) -> list[Voice]:
    if config is None:
        config = VoiceSeparationConfig()
    
    if not events:
        return []
    
    sorted_events = sorted(enumerate(events), key=lambda x: (x[1].start_s, -x[1].note))
    
    chord_groups = chord_grouping(events, config.chord_time_threshold)
    chord_set = set(idx for group in chord_groups for idx in group)
    
    voices: list[Voice] = [Voice()]
    voice_centers: list[float] = [60.0]
    
    for idx, note in sorted_events:
        if idx in chord_set:
            continue
        
        best_voice_idx = 0
        best_cost = float('inf')
        
        for v_idx, voice in enumerate(voices):
            cost = compute_voice_cost(note, voice, config)
            
            center_diff = abs(note.note - voice_centers[v_idx])
            if center_diff > 24:
                cost += center_diff * 0.5
            
            if note.note > 72:
                cost -= config.melody_bias * 10
            
            if cost < best_cost:
                best_cost = cost
                best_voice_idx = v_idx
        
        if best_cost > 50 and len(voices) < config.max_voices:
            voices.append(Voice())
            voice_centers.append(60.0)
            best_voice_idx = len(voices) - 1
        
        voices[best_voice_idx].add_note(note)
        voice_centers[best_voice_idx] = (
            config.hand_smoothing * voice_centers[best_voice_idx] +
            (1 - config.hand_smoothing) * note.note
        )
    
    for group in chord_groups:
        group_notes = [(idx, events[idx]) for idx in group]
        group_notes.sort(key=lambda x: x[1].note)
        
        for sub_idx, (original_idx, note) in enumerate(group_notes):
            if sub_idx < len(voices):
                for v_idx, voice in enumerate(voices):
                    if any(n.note == note.note for n in voice.notes):
                        voices[v_idx].add_note(note)
                        voice_centers[v_idx] = (
                            config.hand_smoothing * voice_centers[v_idx] +
                            (1 - config.hand_smoothing) * note.note
                        )
                        break
            else:
                if len(voices) < config.max_voices:
                    voices.append(Voice())
                    voice_centers.append(60.0)
                best_voice_idx = len(voices) - 1
                voices[best_voice_idx].add_note(note)
                voice_centers[best_voice_idx] = note.note
    
    empty_voices = [i for i, v in enumerate(voices) if not v.notes]
    for i in reversed(empty_voices):
        voices.pop(i)
    
    return voices


def assign_hands(voices: list[Voice]) -> list[int]:
    if not voices:
        return []
    
    voice_scores = []
    for voice in voices:
        score = voice.avg_pitch
        score += 5 * np.std(voice.pitches) if len(voice.pitches) > 1 else 0
        voice_scores.append(score)
    
    hand_assignments = []
    for i, score in enumerate(voice_scores):
        hand_assignments.append(0 if score < 55 else 1)
    
    left_count = sum(1 for h in hand_assignments if h == 0)
    right_count = sum(1 for h in hand_assignments if h == 1)
    
    if left_count == 0 and voices:
        sorted_by_avg = sorted(enumerate(voice_scores), key=lambda x: x[1])
        hand_assignments[sorted_by_avg[0][0]] = 0
        hand_assignments[sorted_by_avg[-1][0]] = 1
    elif right_count == 0 and voices:
        sorted_by_avg = sorted(enumerate(voice_scores), key=lambda x: x[1])
        hand_assignments[sorted_by_avg[0][0]] = 0
        hand_assignments[sorted_by_avg[-1][0]] = 1
    
    return hand_assignments


@dataclass
class VoiceSeparationResult:
    voices: list[Voice]
    hand_assignments: list[int]
    
    def get_notes_for_voice(self, voice_idx: int) -> list[NoteEvent]:
        if 0 <= voice_idx < len(self.voices):
            return self.voices[voice_idx].notes
        return []
    
    def get_left_hand_notes(self) -> list[NoteEvent]:
        left_notes = []
        for v_idx, assignment in enumerate(self.hand_assignments):
            if assignment == 0:
                left_notes.extend(self.voices[v_idx].notes)
        return sorted(left_notes, key=lambda n: n.start_s)
    
    def get_right_hand_notes(self) -> list[NoteEvent]:
        right_notes = []
        for v_idx, assignment in enumerate(self.hand_assignments):
            if assignment == 1:
                right_notes.extend(self.voices[v_idx].notes)
        return sorted(right_notes, key=lambda n: n.start_s)


def separate_voices(
    events: list[NoteEvent],
    config: VoiceSeparationConfig | None = None,
) -> VoiceSeparationResult:
    if config is None:
        config = VoiceSeparationConfig()
    
    voices = assign_voices_dp(events, config)
    hand_assignments = assign_hands(voices)
    
    return VoiceSeparationResult(voices=voices, hand_assignments=hand_assignments)


class VoiceSeparationTranscriber:
    def __init__(
        self,
        base_transcriber,
        config: VoiceSeparationConfig | None = None,
    ):
        self._base = base_transcriber
        self._config = config or VoiceSeparationConfig()
        self.name = f"{base_transcriber.name} + Voice Separation"

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> VoiceSeparationResult:
        events = self._base.transcribe(samples, sample_rate)
        return separate_voices(events, self._config)
    
    def transcribe_to_midi_events(self, samples: np.ndarray, sample_rate: int) -> list[NoteEvent]:
        result = self.transcribe(samples, sample_rate)
        all_notes: list[NoteEvent] = []
        for voice in result.voices:
            all_notes.extend(voice.notes)
        return sorted(all_notes, key=lambda n: (n.start_s, n.note))
