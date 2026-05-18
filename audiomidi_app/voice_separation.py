from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Optional
from copy import deepcopy
import heapq

import numpy as np

from audiomidi_app.midi import NoteEvent


@dataclass(frozen=True)
class VoiceSeparationConfig:
    max_voices: int = 6
    beam_width: int = 8
    
    # Cost weights
    pitch_cost_weight: float = 1.0
    time_cost_weight: float = 0.5
    velocity_cost_weight: float = 0.1
    motion_cost_weight: float = 3.0
    inactivity_cost_weight: float = 0.5
    direction_change_cost_weight: float = 2.0
    
    # Chord grouping
    chord_time_threshold: float = 0.05
    chord_pitch_threshold: float = 24.0
    
    # Voice behavior
    voice_activation_threshold: float = 40.0
    min_note_duration: float = 0.05
    
    # Hand assignment
    hand_smoothing_factor: float = 0.7


@dataclass
class VoiceState:
    id: int
    notes: list[NoteEvent] = field(default_factory=list)
    pitch_history: list[int] = field(default_factory=list)
    time_history: list[float] = field(default_factory=list)
    velocity_history: list[int] = field(default_factory=list)
    pitch_velocity: float = 0.0
    last_pitch: float = 60.0
    last_end_time: float = 0.0
    last_direction: int = 0  # -1, 0, 1
    
    @property
    def avg_pitch(self) -> float:
        if not self.pitch_history:
            return 60.0
        return np.mean(self.pitch_history[-10:])
    
    @property
    def activity_score(self) -> float:
        if not self.notes:
            return 0.0
        recent = [n.end_s for n in self.notes[-3:]]
        return 1.0 / (1.0 + (max(recent) - self.last_end_time) * 0.1)
    
    def predict_next_pitch(self) -> float:
        if len(self.pitch_history) < 2:
            return self.last_pitch
        
        # Linear regression based on last few notes
        recent_pitches = np.array(self.pitch_history[-6:])
        recent_times = np.array(self.time_history[-6:])
        
        if len(recent_pitches) < 3:
            return recent_pitches[-1]
        
        slope, intercept = np.polyfit(recent_times, recent_pitches, 1)
        next_time = recent_times[-1] + 0.2
        return slope * next_time + intercept
    
    def add_note(self, note: NoteEvent) -> None:
        self.notes.append(note)
        self.pitch_history.append(note.note)
        self.time_history.append(note.start_s)
        self.velocity_history.append(note.velocity)
        
        if len(self.pitch_history) >= 2:
            delta = self.pitch_history[-1] - self.pitch_history[-2]
            new_direction = np.sign(delta) if abs(delta) > 1 else 0
            self.pitch_velocity = delta / max(0.01, note.start_s - self.last_end_time)
            self.last_direction = new_direction
        
        self.last_pitch = note.note
        self.last_end_time = note.end_s


@dataclass
class AssignmentHypothesis:
    states: list[VoiceState]
    total_cost: float = 0.0
    n_notes_assigned: int = 0
    
    def __lt__(self, other: AssignmentHypothesis) -> bool:
        return self.total_cost < other.total_cost


def compute_voice_cost(
    note: NoteEvent,
    voice: VoiceState,
    config: VoiceSeparationConfig,
    current_time: float,
) -> float:
    if not voice.notes:
        return 100.0
    
    last_note = voice.notes[-1]
    
    # 1. Pitch continuity
    pitch_diff = abs(note.note - voice.last_pitch)
    pitch_cost = config.pitch_cost_weight * pitch_diff
    
    # 2. Time gap
    time_gap = max(0.0, note.start_s - voice.last_end_time)
    time_cost = config.time_cost_weight * time_gap * 100.0
    
    # 3. Velocity similarity
    velocity_diff = abs(note.velocity - last_note.velocity)
    velocity_cost = config.velocity_cost_weight * velocity_diff
    
    # 4. Motion prediction cost
    predicted_pitch = voice.predict_next_pitch()
    motion_diff = abs(note.note - predicted_pitch)
    motion_cost = config.motion_cost_weight * motion_diff
    
    # 5. Direction change penalty
    direction_cost = 0.0
    if len(voice.pitch_history) >= 2:
        delta = note.note - voice.last_pitch
        new_dir = np.sign(delta) if abs(delta) > 1 else 0
        if voice.last_direction != 0 and new_dir != 0 and new_dir != voice.last_direction:
            direction_cost = config.direction_change_cost_weight * 10.0
    
    # 6. Voice inactivity decay
    inactivity = max(0.0, current_time - voice.last_end_time - 2.0)
    inactivity_cost = config.inactivity_cost_weight * inactivity * 20.0
    
    return pitch_cost + time_cost + velocity_cost + motion_cost + direction_cost + inactivity_cost


def assign_voices_beam_search(
    events: list[NoteEvent],
    config: VoiceSeparationConfig | None = None,
) -> list[VoiceState]:
    if config is None:
        config = VoiceSeparationConfig()
    
    if not events:
        return []
    
    sorted_events = sorted(events, key=lambda n: (n.start_s, -n.note))
    chord_groups = chord_grouping_advanced(sorted_events, config)
    
    beam: list[AssignmentHypothesis] = []
    initial_hypothesis = AssignmentHypothesis(
        states=[VoiceState(id=0)],
        total_cost=0.0,
        n_notes_assigned=0,
    )
    beam.append(initial_hypothesis)
    
    for chord_idx, chord in enumerate(chord_groups):
        current_time = chord[0].start_s if chord else 0.0
        new_beam: list[AssignmentHypothesis] = []
        
        for hypothesis in beam:
            if len(chord) == 1:
                note = chord[0]
                
                # Try assigning to existing voices
                for v_idx, voice in enumerate(hypothesis.states):
                    cost = compute_voice_cost(note, voice, config, current_time)
                    new_states = deepcopy(hypothesis.states)
                    new_states[v_idx].add_note(note)
                    new_hypothesis = AssignmentHypothesis(
                        states=new_states,
                        total_cost=hypothesis.total_cost + cost,
                        n_notes_assigned=hypothesis.n_notes_assigned + 1,
                    )
                    heapq.heappush(new_beam, new_hypothesis)
                    if len(new_beam) > config.beam_width * 2:
                        heapq.heappop(new_beam)
                
                # Try creating new voice
                if len(hypothesis.states) < config.max_voices:
                    new_states = deepcopy(hypothesis.states)
                    new_voice = VoiceState(id=len(new_states))
                    new_voice.add_note(note)
                    new_states.append(new_voice)
                    new_hypothesis = AssignmentHypothesis(
                        states=new_states,
                        total_cost=hypothesis.total_cost + 80.0,
                        n_notes_assigned=hypothesis.n_notes_assigned + 1,
                    )
                    heapq.heappush(new_beam, new_hypothesis)
            
            else:
                assignments = []
                for note in chord:
                    note_assignments = []
                    for v_idx, voice in enumerate(hypothesis.states):
                        cost = compute_voice_cost(note, voice, config, current_time)
                        note_assignments.append((cost, v_idx))
                    
                    if len(hypothesis.states) < config.max_voices:
                        note_assignments.append((80.0, -1))
                    
                    note_assignments.sort()
                    assignments.append(note_assignments)
                
                used_voices = set()
                new_states = deepcopy(hypothesis.states)
                added_cost = 0.0
                
                for note_idx, note in enumerate(chord):
                    assigned = False
                    for cost, v_idx in assignments[note_idx]:
                        if v_idx == -1:
                            if len(new_states) < config.max_voices:
                                new_voice = VoiceState(id=len(new_states))
                                new_voice.add_note(note)
                                new_states.append(new_voice)
                                added_cost += cost
                                assigned = True
                                break
                        elif v_idx not in used_voices:
                            new_states[v_idx].add_note(note)
                            added_cost += cost
                            used_voices.add(v_idx)
                            assigned = True
                            break
                    
                    if not assigned:
                        new_states[0].add_note(note)
                        added_cost += 200.0
                
                new_hypothesis = AssignmentHypothesis(
                    states=new_states,
                    total_cost=hypothesis.total_cost + added_cost,
                    n_notes_assigned=hypothesis.n_notes_assigned + len(chord),
                )
                heapq.heappush(new_beam, new_hypothesis)
        
        beam = heapq.nsmallest(config.beam_width, new_beam)
        if not beam:
            beam = [initial_hypothesis]
    
    if not beam:
        return []
    
    best = min(beam, key=lambda h: h.total_cost)
    result_voices = [s for s in best.states if s.notes]
    
    if not result_voices:
        result_voices = [VoiceState(id=0)]
        for e in sorted_events:
            result_voices[0].add_note(e)
    
    return result_voices


def chord_grouping_advanced(
    events: list[NoteEvent],
    config: VoiceSeparationConfig,
) -> list[list[NoteEvent]]:
    if not events:
        return []
    
    groups: list[list[NoteEvent]] = []
    current_group = [events[0]]
    
    for note in events[1:]:
        last_note = current_group[-1]
        time_diff = note.start_s - last_note.start_s
        
        in_time_range = time_diff <= config.chord_time_threshold
        
        # Compute pitch proximity
        min_pitch = min(n.note for n in current_group)
        max_pitch = max(n.note for n in current_group)
        avg_pitch = np.mean([n.note for n in current_group])
        pitch_spread = max_pitch - min_pitch
        note_pitch_diff = abs(note.note - avg_pitch)
        
        in_pitch_range = (
            note_pitch_diff <= config.chord_pitch_threshold and 
            pitch_spread <= config.chord_pitch_threshold
        )
        
        if in_time_range and in_pitch_range:
            current_group.append(note)
        else:
            groups.append(current_group)
            current_group = [note]
    
    if current_group:
        groups.append(current_group)
    
    return groups


def assign_hands_using_trajectories(
    voices: list[VoiceState],
) -> list[int]:
    if not voices:
        return []
    
    if len(voices) == 1:
        return [1]
    
    sorted_voices = sorted(
        enumerate(voices),
        key=lambda v: (v[1].avg_pitch, len(v[1].notes))
    )
    
    assignments = [1] * len(voices)
    
    n_left = max(1, len(voices) // 2)
    left_voice_indices = [v[0] for v in sorted_voices[:n_left]]
    
    for idx in left_voice_indices:
        assignments[idx] = 0
    
    return assignments


@dataclass
class VoiceSeparationResult:
    voices: list[VoiceState]
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
    
    voices = assign_voices_beam_search(events, config)
    hand_assignments = assign_hands_using_trajectories(voices)
    
    return VoiceSeparationResult(voices=voices, hand_assignments=hand_assignments)
