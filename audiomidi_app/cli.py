from __future__ import annotations

import argparse
from pathlib import Path

from audiomidi_app.audio import read_audio
from audiomidi_app.midi import events_to_midi
from audiomidi_app.transcribe import available_transcribers


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="audiomidi")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--engine", default="Harmonic Salience [DEBUG ONLY]")
    p.add_argument("--bpm", type=float, default=120.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    transcribers = {t.name: t for t in available_transcribers()}
    if args.engine not in transcribers:
        names = ", ".join(transcribers.keys())
        raise SystemExit(f"未知引擎 {args.engine}，可用：{names}")

    audio = read_audio(in_path, target_sr=None, mono=True)
    events = transcribers[args.engine].transcribe(audio.samples, audio.sample_rate)
    mid = events_to_midi(events, bpm=args.bpm)
    mid.save(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
