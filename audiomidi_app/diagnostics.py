from __future__ import annotations

import numpy as np

from audiomidi_app.midi import NoteEvent

_NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
]


def midi_to_note_name(midi_note: int) -> str:
    octave = midi_note // 12 - 1
    name = _NOTE_NAMES[midi_note % 12]
    return f"{name}{octave}"


def print_transcription_report(events: list[NoteEvent], audio_duration: float) -> None:
    if not events:
        print("=== 转谱报告 ===")
        print("  音符数: 0")
        return

    n = len(events)
    density = n / max(audio_duration, 0.001)

    velocities = np.array([e.velocity for e in events], dtype=float)
    durations = np.array([e.end_s - e.start_s for e in events], dtype=float)
    notes = np.array([e.note for e in events], dtype=int)
    confidences = np.array([e.confidence for e in events], dtype=float)

    all_times = []
    for e in events:
        all_times.append(e.start_s)
        all_times.append(e.end_s)
    all_times.sort()

    max_polyphony = 0
    for t in all_times:
        count = sum(1 for e in events if e.start_s <= t < e.end_s)
        if count > max_polyphony:
            max_polyphony = count

    print("=== 转谱报告 ===")
    print(f"  总音符数: {n}")
    print(f"  音符密度: {density:.2f} 音符/秒")
    print(f"  Velocity 分布: min={int(velocities.min())} max={int(velocities.max())} "
          f"mean={velocities.mean():.1f} std={velocities.std():.1f}")
    print(f"  音符时值分布: min={durations.min():.3f}s max={durations.max():.3f}s "
          f"mean={durations.mean():.3f}s")
    print(f"  音域范围: {int(notes.min())} ({midi_to_note_name(int(notes.min()))}) "
          f"~ {int(notes.max())} ({midi_to_note_name(int(notes.max()))})")
    print(f"  最大同时发音数: {max_polyphony}")

    all_one = np.all(confidences == 1.0)
    if not all_one:
        print(f"  Confidence 分布: min={confidences.min():.3f} max={confidences.max():.3f} "
              f"mean={confidences.mean():.3f}")
