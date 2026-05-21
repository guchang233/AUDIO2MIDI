from __future__ import annotations

import argparse
import asyncio
import io
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, FileResponse, HTMLResponse

from audiomidi_app.audio import read_audio
from audiomidi_app.midi import NoteEvent, events_to_midi
from audiomidi_app.transcribe import (
    available_transcribers,
    detect_bpm,
    BatchJobItem,
)

app = FastAPI(title="音频转MIDI", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_jobs: dict[str, JobState] = {}

_frontend_dir: Path | None = None


def mount_frontend(dist_dir: Path) -> None:
    global _frontend_dir
    _frontend_dir = dist_dir.resolve()

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        if full_path.startswith("api/") or full_path.startswith("ws/"):
            return None
        file_path = _frontend_dir / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        index = _frontend_dir / "index.html"
        if index.exists():
            return HTMLResponse(index.read_text(encoding="utf-8"))
        return JSONResponse({"error": "前端未构建"}, status_code=404)


@dataclass
class JobState:
    job_id: str
    status: str = "pending"
    progress: int = 0
    message: str = ""
    stage_text: str = ""
    result_files: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    ws_clients: list[WebSocket] = field(default_factory=list)
    interrupted: bool = False

    async def broadcast(self) -> None:
        payload = {
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "stage_text": self.stage_text,
            "result_files": self.result_files,
            "logs": self.logs,
        }
        dead = []
        for ws in self.ws_clients:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.ws_clients.remove(ws)

    def log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{ts}] {msg}")


@app.get("/api/engines")
async def list_engines():
    engines = []
    for t in available_transcribers():
        engines.append({"name": t.name})
    return JSONResponse(engines)


@app.post("/api/transcribe")
async def transcribe_sync(
    file: UploadFile = File(...),
    engine: str = Form("Piano Transcription (Neural)"),
    bpm: float = Form(120.0),
    auto_bpm: bool = Form(False),
    normalize: bool = Form(True),
    preemphasis: bool = Form(False),
    velocity_stretch: bool = Form(True),
    confidence_threshold: float = Form(0.2),
    bp_onset_threshold: float = Form(0.35),
    bp_frame_threshold: float = Form(0.20),
) -> Response:
    with tempfile.TemporaryDirectory() as td:
        audio_path = Path(td) / (file.filename or "input.wav")
        audio_path.write_bytes(await file.read())

        is_neural = engine in ("Piano Transcription (Neural)", "Basic Pitch", "Ensemble (PT + BP)")
        audio = read_audio(
            str(audio_path),
            target_sr=None,
            mono=True,
            normalize=normalize,
            normalize_mode="rms" if is_neural else "peak",
            preemphasis=preemphasis and not is_neural,
        )

        if auto_bpm:
            bpm = detect_bpm(audio.samples, audio.sample_rate)

        transcribers = available_transcribers()
        transcriber = None
        for t in transcribers:
            if t.name == engine:
                transcriber = t
                break
        if transcriber is None:
            return JSONResponse({"error": f"引擎不可用: {engine}"}, status_code=400)

        if hasattr(transcriber, '_onset_threshold') and hasattr(transcriber, '_frame_threshold'):
            transcriber._onset_threshold = bp_onset_threshold
            transcriber._frame_threshold = bp_frame_threshold
        if hasattr(transcriber, '_bp') and hasattr(transcriber._bp, '_onset_threshold'):
            transcriber._bp._onset_threshold = bp_onset_threshold
            transcriber._bp._frame_threshold = bp_frame_threshold

        events = transcriber.transcribe(audio.samples, audio.sample_rate)

        from audiomidi_app.postprocess import full_postprocess, PostProcessConfig, OnsetDetector
        pp_config = PostProcessConfig(
            confidence_threshold=confidence_threshold,
            enable_velocity_normalize=velocity_stretch,
        )
        onset_detector = OnsetDetector(audio.sample_rate)
        onset_detector.detect(audio.samples)
        events = full_postprocess(
            events,
            samples=audio.samples,
            sample_rate=audio.sample_rate,
            bpm=bpm,
            onset_detector=onset_detector,
            config=pp_config,
            is_neural=is_neural,
        )

        mid = events_to_midi(events, bpm=bpm)
        buf = io.BytesIO()
        mid.save(file=buf)
        return Response(content=buf.getvalue(), media_type="audio/midi")


@app.post("/api/jobs")
async def create_job(
    files: list[UploadFile] = File(...),
    engine: str = Form("Piano Transcription (Neural)"),
    bpm: float = Form(120.0),
    auto_bpm: bool = Form(False),
    normalize: bool = Form(True),
    preemphasis: bool = Form(False),
    velocity_stretch: bool = Form(True),
    confidence_threshold: float = Form(0.2),
    bp_onset_threshold: float = Form(0.35),
    bp_frame_threshold: float = Form(0.20),
) -> JSONResponse:
    job_id = uuid.uuid4().hex[:12]
    state = JobState(job_id=job_id)
    _jobs[job_id] = state

    saved_paths: list[str] = []
    td = tempfile.mkdtemp(prefix="midi_job_")
    for f in files:
        p = Path(td) / (f.filename or "input.wav")
        p.write_bytes(await f.read())
        saved_paths.append(str(p))

    cfg = {
        "engine": engine,
        "bpm": bpm,
        "auto_bpm": auto_bpm,
        "normalize": normalize,
        "preemphasis": preemphasis,
        "velocity_stretch": velocity_stretch,
        "confidence_threshold": confidence_threshold,
        "bp_onset_threshold": bp_onset_threshold,
        "bp_frame_threshold": bp_frame_threshold,
    }

    asyncio.create_task(_run_job(job_id, saved_paths, td, cfg))
    return JSONResponse({"job_id": job_id})


async def _run_job(job_id: str, audio_paths: list[str], work_dir: str, cfg: dict) -> None:
    state = _jobs.get(job_id)
    if state is None:
        return

    engine_name = cfg["engine"]
    is_neural = engine_name in ("Piano Transcription (Neural)", "Basic Pitch", "Ensemble (PT + BP)")
    total = len(audio_paths)
    bpm_val = cfg.get("bpm", 120.0)

    state.status = "running"
    state.progress = 0
    state.message = "正在启动"
    state.stage_text = "正在启动"
    state.log(f"任务初始化...")
    state.log(f"推理引擎: {engine_name}")
    if cfg.get("auto_bpm", False):
        state.log(f"速度(BPM): 自动测速")
    else:
        state.log(f"速度(BPM): {bpm_val}")
    await state.broadcast()

    transcribers = available_transcribers()
    transcriber = None
    for t in transcribers:
        if t.name == engine_name:
            transcriber = t
            break
    if transcriber is None:
        state.status = "failed"
        state.message = f"引擎不可用: {engine_name}"
        await state.broadcast()
        return

    if hasattr(transcriber, '_onset_threshold') and hasattr(transcriber, '_frame_threshold'):
        transcriber._onset_threshold = cfg.get("bp_onset_threshold", 0.35)
        transcriber._frame_threshold = cfg.get("bp_frame_threshold", 0.20)
    if hasattr(transcriber, '_bp') and hasattr(transcriber._bp, '_onset_threshold'):
        transcriber._bp._onset_threshold = cfg.get("bp_onset_threshold", 0.35)
        transcriber._bp._frame_threshold = cfg.get("bp_frame_threshold", 0.20)

    from audiomidi_app.postprocess import full_postprocess, PostProcessConfig, OnsetDetector

    out_dir = Path(work_dir) / "output"
    out_dir.mkdir(exist_ok=True)

    for idx, audio_path in enumerate(audio_paths):
        if state.interrupted:
            state.status = "cancelled"
            state.message = "已取消"
            await state.broadcast()
            return

        fname = Path(audio_path).name
        out_name = Path(audio_path).with_suffix(".mid").name

        def set_stage(pct: int, stage: str) -> None:
            if total <= 1:
                state.progress = pct
            else:
                base = int(idx / total * 100)
                file_share = int(95 / total)
                state.progress = min(base + int(pct * file_share / 100), 99)
            state.stage_text = stage
            asyncio.create_task(state.broadcast())

        set_stage(0, f"[{idx + 1}/{total}] 正在启动")
        state.log(f"输入文件: {audio_path}")
        state.log(f"输出目录: {out_dir}")
        state.log(f"导出目标: {out_name}")
        await state.broadcast()

        try:
            set_stage(5, f"[{idx + 1}/{total}] 分析音频 (1/4)")
            state.log("分析音频 (1/4)")
            audio = read_audio(
                audio_path,
                target_sr=None,
                mono=True,
                normalize=cfg.get("normalize", True),
                normalize_mode="rms" if is_neural else "peak",
                preemphasis=cfg.get("preemphasis", False) and not is_neural,
            )
            duration = len(audio.samples) / audio.sample_rate
            state.log(f"✅ 音频序列加载就绪 | 采样率: {audio.sample_rate}Hz | 总长: {duration:.2f}秒")
            await state.broadcast()

            if state.interrupted:
                return

            bpm = bpm_val
            if cfg.get("auto_bpm", False):
                bpm = detect_bpm(audio.samples, audio.sample_rate)
                state.log(f"✅ BPM检测完成: {bpm:.1f}")

            set_stage(35, f"[{idx + 1}/{total}] 模型推理中 (2/4) - {engine_name}")
            state.log(f"模型推理中 (2/4) - {engine_name}")
            await state.broadcast()

            def on_segment_progress(seg_idx: int, total_segments: int) -> None:
                if total_segments <= 1:
                    return
                seg_pct = 35 + int((seg_idx + 1) / total_segments * 15)
                set_stage(seg_pct, f"模型特征解码中 (2/4) - {engine_name} [{seg_idx + 1}/{total_segments}]")
                state.log(f"Segment {seg_idx + 1} / {total_segments}")

            events = transcriber.transcribe(
                audio.samples, audio.sample_rate,
                progress_callback=on_segment_progress,
                interrupt_check=lambda: state.interrupted,
            )

            n_notes = len(events)
            state.log(f"✅ 音符检测完成: {n_notes} 个")
            set_stage(50, f"[{idx + 1}/{total}] 执行数据后处理优化 (3/5)")
            state.log("执行数据后处理优化 (3/5)")
            await state.broadcast()

            pp_config = PostProcessConfig(
                confidence_threshold=cfg.get("confidence_threshold", 0.2),
                enable_velocity_normalize=cfg.get("velocity_stretch", True),
            )
            onset_detector = OnsetDetector(audio.sample_rate)
            onset_detector.detect(audio.samples)
            events = full_postprocess(
                events,
                samples=audio.samples,
                sample_rate=audio.sample_rate,
                bpm=bpm,
                onset_detector=onset_detector,
                config=pp_config,
                is_neural=is_neural,
            )

            n_after_pp = len(events)
            state.log(f"✅ 过滤后处理完成 | 有效音符留存: {n_after_pp}")
            set_stage(60, f"[{idx + 1}/{total}] 生成转谱报告")

            try:
                from audiomidi_app.diagnostics import print_transcription_report
                import io as _io, sys
                buf = _io.StringIO()
                old_stdout = sys.stdout
                sys.stdout = buf
                try:
                    print_transcription_report(events, duration)
                finally:
                    sys.stdout = old_stdout
                for line in buf.getvalue().strip().split("\n"):
                    if line.strip():
                        state.log(line)
            except Exception:
                pass

            use_voice_sep = cfg.get("use_voice_separation", False)
            split_hands = cfg.get("split_hands", True)
            left_ch = cfg.get("left_hand_channel", 1)
            right_ch = cfg.get("right_hand_channel", 2)

            voice_result = None
            if use_voice_sep:
                set_stage(70, f"[{idx + 1}/{total}] 声部分离中 (3/4)")
                state.log("声部分离中 (3/4)")
                try:
                    from audiomidi_app.voice_separation import separate_voices, VoiceSeparationResult
                    voice_result = separate_voices(events)
                    n_voices = len(voice_result.voices) if voice_result else 0
                    state.log(f"✅ 声部分离完成 | 音轨数: {n_voices}")
                except Exception as e:
                    state.log(f"⚠️ 声部分离失败: {e}")
                set_stage(75, f"[{idx + 1}/{total}] 导出MIDI (4/4)")
            else:
                set_stage(85, f"[{idx + 1}/{total}] 导出MIDI (4/4)")

            state.log("导出MIDI (4/4)")
            await state.broadcast()

            out_path = out_dir / out_name
            if voice_result and split_hands:
                from audiomidi_app.midi import events_to_midi_with_hands
                mid = events_to_midi_with_hands(
                    voice_result, bpm=bpm,
                    left_channel=left_ch,
                    right_channel=right_ch,
                )
            else:
                mid = events_to_midi(events, bpm=bpm)

            mid.save(str(out_path))
            size_kb = out_path.stat().st_size / 1024
            state.result_files.append(out_name)
            state.log(f"✅ 编译成功: {out_name} ({size_kb:.1f} KB)")
            set_stage(95, f"[{idx + 1}/{total}] 完成")

        except Exception as e:
            state.log(f"❌ 失败: {fname} - {e}")

        await state.broadcast()

    state.status = "done"
    state.progress = 100
    state.message = f"{len(state.result_files)}/{total} 完成"
    state.stage_text = "任务完成"
    state.log(f"====== 任务完成 ======")
    await state.broadcast()


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str) -> JSONResponse:
    state = _jobs.get(job_id)
    if state is None:
        return JSONResponse({"error": "任务不存在"}, status_code=404)
    return JSONResponse({
        "job_id": state.job_id,
        "status": state.status,
        "progress": state.progress,
        "message": state.message,
        "stage_text": state.stage_text,
        "result_files": state.result_files,
        "logs": state.logs,
    })


@app.websocket("/ws/{job_id}")
async def ws_job(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    state = _jobs.get(job_id)
    if state is None:
        await websocket.send_json({"error": "任务不存在"})
        await websocket.close()
        return

    state.ws_clients.append(websocket)
    try:
        payload = {
            "status": state.status,
            "progress": state.progress,
            "message": state.message,
            "stage_text": state.stage_text,
            "result_files": state.result_files,
            "logs": state.logs,
        }
        await websocket.send_json(payload)
        while True:
            data = await websocket.receive_text()
            if data == "cancel":
                state.interrupted = True
                state.status = "cancelling"
                await state.broadcast()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in state.ws_clients:
            state.ws_clients.remove(websocket)


@app.get("/api/jobs/{job_id}/download/{filename}")
async def download_result(job_id: str, filename: str) -> Response:
    state = _jobs.get(job_id)
    if state is None:
        return JSONResponse({"error": "任务不存在"}, status_code=404)
    if filename not in state.result_files:
        return JSONResponse({"error": "文件不存在"}, status_code=404)

    work_dir = Path(tempfile.gettempdir()) / f"midi_job_{job_id}" / "output"
    file_path = work_dir / filename
    if not file_path.exists():
        return JSONResponse({"error": "文件已过期"}, status_code=404)

    return Response(
        content=file_path.read_bytes(),
        media_type="audio/midi",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="音频转MIDI Web服务")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--frontend", nargs="?", const="web/dist", default=None,
                    help="挂载前端静态文件目录（默认 web/dist），不指定则纯API模式")
    args = p.parse_args(argv)

    if args.frontend:
        dist = Path(args.frontend).resolve()
        if dist.exists() and dist.is_dir():
            mount_frontend(dist)
            print(f"前端静态文件托管: {dist}")
        else:
            print(f"警告: 前端目录不存在: {dist}，仅启动 API 模式")

    print(f"音频转MIDI Web服务启动: http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
