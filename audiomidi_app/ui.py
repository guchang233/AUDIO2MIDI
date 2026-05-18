from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audiomidi_app.audio import read_audio
from audiomidi_app.cloud_client import CloudConfig, transcribe_via_cloud
from audiomidi_app.midi import NoteEvent, events_to_midi
from audiomidi_app.transcribe import (
    available_transcribers,
    try_basic_pitch_transcriber,
    available_voice_separation_transcribers,
    VoiceSeparationTranscriber,
)
from audiomidi_app.voice_separation import separate_voices, VoiceSeparationResult

_qt_import_error: Exception | None = None
try:
    from PySide6.QtCore import QObject, Qt, QThread, Signal, QTimer
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QProgressBar,
        QPushButton,
        QSpinBox,
        QTabWidget,
        QVBoxLayout,
        QWidget,
        QGroupBox,
        QScrollArea,
    )
    from PySide6.QtGui import QDragEnterEvent, QDropEvent, QFont
except Exception as e:
    _qt_import_error = e


@dataclass(frozen=True)
class JobConfig:
    audio_path: str
    out_path: str
    engine: str
    bpm: float
    cloud_enabled: bool
    cloud_base_url: str
    use_voice_separation: bool
    split_hands: bool
    left_hand_channel: int
    right_hand_channel: int


def run_app() -> None:
    if _qt_import_error is not None:
        raise RuntimeError(f"桌面UI依赖加载失败：{_qt_import_error}")

    class Worker(QObject):
        progress = Signal(str)
        done = Signal(str)
        failed = Signal(str)
        _interrupted = False
        _result = None

        def __init__(self, cfg: JobConfig) -> None:
            super().__init__()
            self._cfg = cfg
            self._interrupted = False

        def interrupt(self) -> None:
            self._interrupted = True

        def run(self) -> None:
            try:
                self.progress.emit("开始转谱")
                out = self._run_impl()
                if self._interrupted:
                    return
                self.done.emit(out)
            except Exception as e:
                if not self._interrupted:
                    self.failed.emit(str(e))

        def _run_impl(self) -> str:
            audio_path = Path(self._cfg.audio_path)
            out_path = Path(self._cfg.out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if self._interrupted:
                return ""

            if self._cfg.cloud_enabled:
                try:
                    self.progress.emit("调用云端转谱")
                    midi_bytes = transcribe_via_cloud(
                        CloudConfig(base_url=self._cfg.cloud_base_url),
                        audio_path=audio_path,
                        engine=self._cfg.engine,
                        bpm=self._cfg.bpm,
                    )
                    out_path.write_bytes(midi_bytes)
                    return str(out_path)
                except Exception as e:
                    self.progress.emit(f"云端失败，回退本地：{e}")

            if self._interrupted:
                return ""

            if self._cfg.engine == "Basic Pitch":
                self.progress.emit("运行 Basic Pitch")
                bp = try_basic_pitch_transcriber()
                if bp is None or not hasattr(bp, "transcribe_file"):
                    raise RuntimeError("当前环境未安装 basic-pitch 或不兼容")
                midi_path = bp.transcribe_file(str(audio_path), out_dir=str(out_path.parent))
                if Path(midi_path) != out_path:
                    out_path.write_bytes(Path(midi_path).read_bytes())
                return str(out_path)

            self.progress.emit("分析音频")
            audio = read_audio(audio_path, target_sr=None, mono=True)

            if self._interrupted:
                return ""

            self.progress.emit(f"生成音符（引擎：{self._cfg.engine}）")
            
            transcribers = available_transcribers()
            transcriber: Any = None
            for t in transcribers:
                if t.name == self._cfg.engine:
                    transcriber = t
                    break
            
            if transcriber is None:
                raise RuntimeError(f"找不到引擎：{self._cfg.engine}")

            if self._interrupted:
                return ""

            events = transcriber.transcribe(audio.samples, audio.sample_rate)
            voice_result: VoiceSeparationResult | None = None

            if self._cfg.use_voice_separation:
                self.progress.emit("声部分离中...")
                voice_result = separate_voices(events)

            self.progress.emit("写入MIDI")
            
            if voice_result and self._cfg.split_hands:
                self.progress.emit("分离左右手...")
                mid = events_to_midi_with_hands(
                    voice_result, 
                    bpm=self._cfg.bpm,
                    left_channel=self._cfg.left_hand_channel,
                    right_channel=self._cfg.right_hand_channel,
                )
            else:
                mid = events_to_midi(events, bpm=self._cfg.bpm)
            
            mid.save(str(out_path))
            return str(out_path)

    class MainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle("Audio → MIDI (高级版)")
            self.setAcceptDrops(True)
            self._setup_ui()

            self._thread: QThread | None = None
            self._worker: Worker | None = None

        def _setup_ui(self) -> None:
            root = QWidget()
            self.setCentralWidget(root)
            main_layout = QVBoxLayout(root)

            # 顶部：文件选择区域
            file_group = QGroupBox("文件")
            file_layout = QFormLayout()

            self._audio_path = QLineEdit()
            self._audio_path.setReadOnly(True)
            self._audio_path.setPlaceholderText("拖拽音频文件到此处或点击选择")
            pick_audio = QPushButton("选择音频")
            pick_audio.clicked.connect(self._on_pick_audio)
            row_audio = QHBoxLayout()
            row_audio.addWidget(self._audio_path, 1)
            row_audio.addWidget(pick_audio)
            file_layout.addRow("音频输入", row_audio)

            self._out_path = QLineEdit()
            self._out_path.setReadOnly(True)
            pick_out = QPushButton("选择输出")
            pick_out.clicked.connect(self._on_pick_out)
            row_out = QHBoxLayout()
            row_out.addWidget(self._out_path, 1)
            row_out.addWidget(pick_out)
            file_layout.addRow("MIDI输出", row_out)

            file_group.setLayout(file_layout)
            main_layout.addWidget(file_group)

            # 标签页
            tabs = QTabWidget()

            # 转谱设置
            tab_transcribe = QWidget()
            transcribe_layout = QFormLayout(tab_transcribe)

            self._engine = QComboBox()
            for t in available_transcribers():
                self._engine.addItem(t.name)
            self._engine.currentIndexChanged.connect(self._on_engine_changed)
            transcribe_layout.addRow("转谱引擎", self._engine)

            self._bpm = QDoubleSpinBox()
            self._bpm.setRange(30.0, 400.0)
            self._bpm.setSingleStep(1.0)
            self._bpm.setDecimals(2)
            self._bpm.setValue(120.0)
            transcribe_layout.addRow("BPM", self._bpm)

            self._auto_bpm = QCheckBox("自动检测BPM")
            transcribe_layout.addRow("", self._auto_bpm)

            tabs.addTab(tab_transcribe, "转谱")

            # 声部分离
            tab_voice = QWidget()
            voice_layout = QVBoxLayout(tab_voice)

            self._use_voice_sep = QCheckBox("启用声部分离")
            self._use_voice_sep.setChecked(False)
            self._use_voice_sep.stateChanged.connect(self._on_voice_sep_toggled)
            voice_layout.addWidget(self._use_voice_sep)

            voice_options = QGroupBox("声部分离选项")
            voice_options_layout = QFormLayout()
            voice_options_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)

            self._split_hands = QCheckBox("左右手分离")
            self._split_hands.setEnabled(False)
            self._split_hands.setChecked(True)
            voice_options_layout.addRow("", self._split_hands)

            self._left_channel = QSpinBox()
            self._left_channel.setRange(1, 16)
            self._left_channel.setValue(1)
            self._left_channel.setEnabled(False)
            voice_options_layout.addRow("左手Channel", self._left_channel)

            self._right_channel = QSpinBox()
            self._right_channel.setRange(1, 16)
            self._right_channel.setValue(2)
            self._right_channel.setEnabled(False)
            voice_options_layout.addRow("右手Channel", self._right_channel)

            voice_options.setLayout(voice_options_layout)
            voice_layout.addWidget(voice_options)
            voice_layout.addStretch()

            tabs.addTab(tab_voice, "声部分离")

            # 云端选项
            tab_cloud = QWidget()
            cloud_layout = QFormLayout(tab_cloud)

            self._cloud = QCheckBox("云端优先（失败自动回退本地）")
            self._cloud.stateChanged.connect(self._on_cloud_toggled)
            cloud_layout.addRow("", self._cloud)

            self._cloud_url = QLineEdit("http://127.0.0.1:8000")
            self._cloud_url.setEnabled(False)
            cloud_layout.addRow("云端地址", self._cloud_url)

            tabs.addTab(tab_cloud, "云端")

            main_layout.addWidget(tabs, 1)

            # 按钮
            btn_row = QHBoxLayout()
            self._run = QPushButton("开始转谱")
            self._run.setMinimumHeight(40)
            font = self._run.font()
            font.setPointSize(11)
            font.setBold(True)
            self._run.setFont(font)
            self._run.clicked.connect(self._on_run)
            btn_row.addWidget(self._run)

            self._stop = QPushButton("停止")
            self._stop.setEnabled(False)
            self._stop.setMinimumHeight(40)
            self._stop.clicked.connect(self._on_stop)
            btn_row.addWidget(self._stop)
            main_layout.addLayout(btn_row)

            # 进度
            self._progress = QProgressBar()
            self._progress.setVisible(False)
            main_layout.addWidget(self._progress)

            # 状态栏
            status_group = QGroupBox("状态")
            status_layout = QVBoxLayout()

            self._status = QLabel("就绪")
            self._status.setTextInteractionFlags(Qt.TextSelectableByMouse)
            status_layout.addWidget(self._status)

            self._stats = QLabel("")
            self._stats.setTextInteractionFlags(Qt.TextSelectableByMouse)
            status_layout.addWidget(self._stats)

            status_group.setLayout(status_layout)
            main_layout.addWidget(status_group)

        def dragEnterEvent(self, event: QDragEnterEvent) -> None:
            if event.mimeData().hasUrls():
                for url in event.mimeData().urls():
                    if url.isLocalFile():
                        path = url.toLocalFile()
                        ext = Path(path).suffix.lower()
                        if ext in ['.wav', '.flac', '.ogg', '.mp3', '.m4a']:
                            event.acceptProposedAction()
                            return
            event.ignore()

        def dropEvent(self, event: QDropEvent) -> None:
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    path = url.toLocalFile()
                    ext = Path(path).suffix.lower()
                    if ext in ['.wav', '.flac', '.ogg', '.mp3', '.m4a']:
                        self._audio_path.setText(path)
                        if not self._out_path.text():
                            self._out_path.setText(str(Path(path).with_suffix(".mid")))
                        return

        def _on_engine_changed(self, idx: int) -> None:
            pass

        def _on_voice_sep_toggled(self, state: int) -> None:
            enabled = state == Qt.Checked.value
            self._split_hands.setEnabled(enabled)
            self._left_channel.setEnabled(enabled and self._split_hands.isChecked())
            self._right_channel.setEnabled(enabled and self._split_hands.isChecked())

        def _on_cloud_toggled(self, state: int) -> None:
            self._cloud_url.setEnabled(state == Qt.Checked.value)

        def _on_pick_audio(self) -> None:
            path, _ = QFileDialog.getOpenFileName(
                self, 
                "选择音频", 
                "", 
                "Audio (*.wav *.flac *.ogg *.mp3 *.m4a);;All (*)"
            )
            if not path:
                return
            self._audio_path.setText(path)
            if not self._out_path.text():
                self._out_path.setText(str(Path(path).with_suffix(".mid")))

        def _on_pick_out(self) -> None:
            path, _ = QFileDialog.getSaveFileName(self, "选择输出MIDI", "", "MIDI (*.mid)")
            if not path:
                return
            if not path.lower().endswith(".mid"):
                path += ".mid"
            self._out_path.setText(path)

        def _set_ui_running(self, running: bool) -> None:
            self._run.setEnabled(not running)
            self._stop.setEnabled(running)
            self._engine.setEnabled(not running)
            self._bpm.setEnabled(not running)
            self._cloud.setEnabled(not running)
            self._cloud_url.setEnabled(not running and self._cloud.isChecked())
            self._audio_path.setEnabled(not running)
            self._out_path.setEnabled(not running)
            self._use_voice_sep.setEnabled(not running)
            self._split_hands.setEnabled(not running and self._use_voice_sep.isChecked())
            self._left_channel.setEnabled(not running and self._use_voice_sep.isChecked() and self._split_hands.isChecked())
            self._right_channel.setEnabled(not running and self._use_voice_sep.isChecked() and self._split_hands.isChecked())
            self._progress.setVisible(running)

        def _on_run(self) -> None:
            if self._thread is not None:
                return

            audio = self._audio_path.text().strip()
            outp = self._out_path.text().strip()
            if not audio or not outp:
                self._status.setText("请选择音频与输出路径")
                return

            if not Path(audio).exists():
                self._status.setText("音频文件不存在")
                return

            cfg = JobConfig(
                audio_path=audio,
                out_path=outp,
                engine=self._engine.currentText(),
                bpm=self._bpm.value(),
                cloud_enabled=self._cloud.isChecked(),
                cloud_base_url=self._cloud_url.text().strip(),
                use_voice_separation=self._use_voice_sep.isChecked(),
                split_hands=self._split_hands.isChecked(),
                left_hand_channel=self._left_channel.value(),
                right_hand_channel=self._right_channel.value(),
            )

            self._set_ui_running(True)
            self._status.setText("准备中...")
            self._stats.setText("")

            self._thread = QThread()
            self._worker = Worker(cfg)
            self._worker.moveToThread(self._thread)
            self._thread.started.connect(self._worker.run)
            self._worker.progress.connect(self._status.setText)
            self._worker.done.connect(self._on_done)
            self._worker.failed.connect(self._on_failed)
            self._worker.done.connect(self._thread.quit)
            self._worker.failed.connect(self._thread.quit)
            self._thread.finished.connect(self._cleanup_thread)
            self._thread.start()

        def _on_stop(self) -> None:
            if self._worker is not None:
                self._worker.interrupt()
            self._status.setText("正在停止...")

        def _on_done(self, out_path: str) -> None:
            if out_path:
                self._status.setText(f"✅ 完成：{out_path}")
            else:
                self._status.setText("⏹ 已停止")

        def _on_failed(self, msg: str) -> None:
            self._status.setText(f"❌ 失败：{msg}")

        def _cleanup_thread(self) -> None:
            self._set_ui_running(False)
            
            if self._worker is not None:
                self._worker.deleteLater()
            if self._thread is not None:
                self._thread.deleteLater()
            self._thread = None
            self._worker = None

    app = QApplication([])
    w = MainWindow()
    w.resize(720, 520)
    w.show()
    app.exec()


def events_to_midi_with_hands(
    voice_result: VoiceSeparationResult,
    bpm: float = 120.0,
    left_channel: int = 1,
    right_channel: int = 2,
) -> Any:
    try:
        import mido
        from mido import MidiFile, MidiTrack
    except ImportError:
        raise RuntimeError("mido 未安装")

    mid = MidiFile()

    left_notes = voice_result.get_left_hand_notes()
    right_notes = voice_result.get_right_hand_notes()

    if left_notes:
        left_track = MidiTrack()
        mid.tracks.append(left_track)
        _append_notes_to_track(left_track, left_notes, bpm, channel=left_channel - 1)

    if right_notes:
        right_track = MidiTrack()
        mid.tracks.append(right_track)
        _append_notes_to_track(right_track, right_notes, bpm, channel=right_channel - 1)

    return mid


def _append_notes_to_track(track, events: list[NoteEvent], bpm: float, channel: int) -> None:
    import mido
    sorted_events = sorted(events, key=lambda n: n.start_s)
    us_per_beat = 60_000_000 / bpm
    ticks_per_beat = 480

    track.append(mido.MetaMessage('set_tempo', tempo=int(us_per_beat)))
    track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4))

    messages = []
    for e in sorted_events:
        on_time = int(e.start_s * (ticks_per_beat * bpm / 60))
        off_time = int(e.end_s * (ticks_per_beat * bpm / 60))
        messages.append((on_time, 'note_on', e.note, e.velocity))
        messages.append((off_time, 'note_off', e.note, e.velocity))

    messages.sort(key=lambda x: x[0])
    last_time = 0
    for time, msg_type, note, velocity in messages:
        dt = time - last_time
        if dt < 0:
            dt = 0
        if msg_type == 'note_on':
            track.append(mido.Message('note_on', note=note, velocity=velocity, time=dt, channel=channel))
        else:
            track.append(mido.Message('note_off', note=note, velocity=0, time=dt, channel=channel))
        last_time = time
