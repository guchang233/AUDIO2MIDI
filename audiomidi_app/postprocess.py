from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from audiomidi_app.midi import NoteEvent


@dataclass(frozen=True)
class PostProcessConfig:
    min_note_gap_s: float = 0.05
    min_note_duration_s: float = 0.05
    onset_refinement_threshold_s: float = 0.01
    repeated_note_max_gap_s: float = 0.15
    velocity_smoothing_window: int = 3
    velocity_calibration_mode: str = "piano"
    
    enable_quantization: bool = True
    quantize_division: float = 0.25


def detect_repeated_notes(
    events: list[NoteEvent],
    config: PostProcessConfig | None = None,
) -> list[NoteEvent]:
    if config is None:
        config = PostProcessConfig()
    
    if not events:
        return events
    
    events = sorted(events, key=lambda e: (e.start_s, e.note))
    result: list[NoteEvent] = []
    
    by_note: dict[int, list[NoteEvent]] = {}
    for e in events:
        by_note.setdefault(e.note, []).append(e)
    
    for note_num, note_events in by_note.items():
        note_events.sort(key=lambda e: e.start_s)
        
        active: list[NoteEvent] = []
        
        for note in note_events:
            should_split = False
            
            for active_note in active:
                gap = note.start_s - active_note.end_s
                if gap < 0:
                    overlap = active_note.end_s - note.start_s
                    if overlap > 0.02:
                        should_split = True
                        break
                elif gap <= config.min_note_gap_s:
                    gap_to_prev_end = note.start_s - active_note.end_s
                    if gap_to_prev_end > config.repeated_note_max_gap_s:
                        should_split = True
                        break
                    gap_ratio = gap_to_prev_end / max(0.01, active_note.end_s - active_note.start_s)
                    if gap_ratio > 0.3:
                        should_split = True
                        break
                else:
                    break
            
            active = [n for n in active if n.end_s > note.start_s]
            
            if should_split:
                result.append(note)
                active.append(note)
            else:
                if active:
                    merged = active[-1]
                    merged = NoteEvent(
                        note=note.note,
                        start_s=merged.start_s,
                        end_s=max(merged.end_s, note.end_s),
                        velocity=max(merged.velocity, note.velocity),
                    )
                    active[-1] = merged
                else:
                    result.append(note)
                    active.append(note)
        
        for active_note in active:
            if active_note not in result:
                result.append(active_note)
    
    result.sort(key=lambda e: (e.start_s, e.note))
    return result


def refine_onsets_with_audio(
    events: list[NoteEvent],
    samples: np.ndarray,
    sample_rate: int,
    config: PostProcessConfig | None = None,
) -> list[NoteEvent]:
    if config is None:
        config = PostProcessConfig()
    
    if not events:
        return events
    
    try:
        import librosa
    except ImportError:
        return events
    
    onset_strength = librosa.onset.onset_strength(y=samples, sr=sample_rate)
    frames = librosa.frames_to_time(
        np.arange(len(onset_strength)),
        sr=sample_rate,
        hop_length=512
    )
    
    refined_events: list[NoteEvent] = []
    
    for event in events:
        onset_time = event.start_s
        
        frame_idx = np.searchsorted(frames, onset_time)
        
        search_start = max(0, frame_idx - 5)
        search_end = min(len(onset_strength), frame_idx + 5)
        
        if search_start < search_end:
            local_onset = onset_strength[search_start:search_end]
            
            peaks_idx = []
            for i in range(1, len(local_onset) - 1):
                if local_onset[i] > local_onset[i-1] and local_onset[i] > local_onset[i+1]:
                    peaks_idx.append(i)
            
            if peaks_idx:
                peak_frame = search_start + max(peaks_idx, key=lambda i: local_onset[i])
                refined_onset = frames[peak_frame]
                
                if abs(refined_onset - onset_time) <= config.onset_refinement_threshold_s:
                    onset_time = refined_onset
        
        refined_events.append(NoteEvent(
            note=event.note,
            start_s=onset_time,
            end_s=event.end_s,
            velocity=event.velocity,
        ))
    
    return refined_events


def split_repeated_notes_by_spectral_flux(
    events: list[NoteEvent],
    samples: np.ndarray,
    sample_rate: int,
    config: PostProcessConfig | None = None,
) -> list[NoteEvent]:
    if config is None:
        config = PostProcessConfig()
    
    if not events:
        return events
    
    try:
        import librosa
    except ImportError:
        return events
    
    onset_frames = librosa.onset.onset_detect(
        y=samples,
        sr=sample_rate,
        hop_length=512,
        backtrack=True,
        units="time"
    )
    
    onset_set = set(np.round(onset_frames, 3))
    
    result: list[NoteEvent] = []
    
    events_by_note: dict[int, list[NoteEvent]] = {}
    for e in events:
        events_by_note.setdefault(e.note, []).append(e)
    
    for note_num, note_events in events_by_note.items():
        note_events.sort(key=lambda e: e.start_s)
        
        current: Optional[NoteEvent] = None
        
        for note in note_events:
            if current is None:
                current = note
                continue
            
            gap = note.start_s - current.end_s
            
            if gap > 0 and gap <= config.repeated_note_max_gap_s:
                nearby_onset = False
                for onset_t in onset_set:
                    if abs(onset_t - note.start_s) <= 0.05:
                        nearby_onset = True
                        break
                
                if nearby_onset:
                    result.append(current)
                    current = note
                    continue
            
            current = NoteEvent(
                note=note.note,
                start_s=current.start_s,
                end_s=max(current.end_s, note.end_s),
                velocity=max(current.velocity, note.velocity),
            )
        
        if current is not None:
            result.append(current)
    
    result.sort(key=lambda e: (e.start_s, e.note))
    return result


def calibrate_velocity(
    events: list[NoteEvent],
    mode: str = "piano",
) -> list[NoteEvent]:
    if not events:
        return events
    
    if mode == "piano":
        calibrated: list[NoteEvent] = []
        
        for event in events:
            velocity = event.velocity
            
            db = 20.0 * np.log10(max(1e-9, velocity / 127.0))
            
            db_calibrated = db + 3.0
            
            velocity_calibrated = int(np.clip(127.0 * (10 ** (db_calibrated / 20.0)), 1, 127))
            
            calibrated.append(NoteEvent(
                note=event.note,
                start_s=event.start_s,
                end_s=event.end_s,
                velocity=velocity_calibrated,
            ))
        
        return calibrated
    
    return events


def smooth_velocities(
    events: list[NoteEvent],
    window_size: int = 3,
) -> list[NoteEvent]:
    if not events or window_size < 2:
        return events
    
    events = sorted(events, key=lambda e: e.start_s)
    
    by_note: dict[int, list[NoteEvent]] = {}
    for e in events:
        by_note.setdefault(e.note, []).append(e)
    
    result: list[NoteEvent] = []
    
    for note_num, note_events in by_note.items():
        for i, event in enumerate(note_events):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(note_events), i + window_size // 2 + 1)
            
            window_velocities = [n.velocity for n in note_events[start_idx:end_idx]]
            smoothed_velocity = int(round(np.median(window_velocities)))
            
            result.append(NoteEvent(
                note=event.note,
                start_s=event.start_s,
                end_s=event.end_s,
                velocity=smoothed_velocity,
            ))
    
    result.sort(key=lambda e: (e.start_s, e.note))
    return result


def quantize_onsets(
    events: list[NoteEvent],
    bpm: float = 120.0,
    division: float = 0.25,
) -> list[NoteEvent]:
    if not events or division <= 0:
        return events
    
    beat_duration = 60.0 / bpm
    grid_duration = beat_duration * division
    
    quantized: list[NoteEvent] = []
    
    for event in events:
        nearest_grid = round(event.start_s / grid_duration) * grid_duration
        
        if abs(event.start_s - nearest_grid) <= grid_duration * 0.3:
            start_s = nearest_grid
        else:
            start_s = event.start_s
        
        quantized.append(NoteEvent(
            note=event.note,
            start_s=start_s,
            end_s=event.end_s,
            velocity=event.velocity,
        ))
    
    return quantized


def full_postprocess(
    events: list[NoteEvent],
    samples: np.ndarray | None = None,
    sample_rate: int = 44100,
    bpm: float = 120.0,
    config: PostProcessConfig | None = None,
) -> list[NoteEvent]:
    if config is None:
        config = PostProcessConfig()
    
    if not events:
        return events
    
    events = detect_repeated_notes(events, config)
    
    if samples is not None:
        events = refine_onsets_with_audio(events, samples, sample_rate, config)
        events = split_repeated_notes_by_spectral_flux(events, samples, sample_rate, config)
    
    events = [NoteEvent(
        note=e.note,
        start_s=max(0, e.start_s),
        end_s=max(e.start_s + config.min_note_duration_s, e.end_s),
        velocity=max(1, min(127, e.velocity)),
    ) for e in events]
    
    events = calibrate_velocity(events, config.velocity_calibration_mode)
    
    if config.enable_quantization:
        events = quantize_onsets(events, bpm, config.quantize_division)
    
    events.sort(key=lambda e: (e.start_s, e.note))
    
    return events


def estimate_note_confidence(
    event: NoteEvent,
    all_events: list[NoteEvent],
) -> float:
    same_note = [e for e in all_events if e.note == event.note]
    
    confidence = 1.0
    
    duration = event.end_s - event.start_s
    if duration < 0.1:
        confidence *= 0.8
    elif duration > 5.0:
        confidence *= 0.9
    
    if same_note:
        gaps = []
        sorted_same = sorted(same_note, key=lambda e: e.start_s)
        for i in range(len(sorted_same) - 1):
            gap = sorted_same[i + 1].start_s - sorted_same[i].end_s
            gaps.append(gap)
        
        if gaps:
            avg_gap = np.mean(gaps)
            if avg_gap > 0.5:
                confidence *= 1.2
            elif avg_gap < 0.1:
                confidence *= 0.7
    
    return min(1.0, max(0.0, confidence))
