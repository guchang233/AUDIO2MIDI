
# Audio to MIDI Transcription System

音频转 MIDI 工具 - 提供基于 DSP 的精确转录功能。

## 功能说明

本项目提供两种 DSP 音频转 MIDI 引擎：

- **Harmonic Salience** - 谐波显著性检测（推荐，效果较好）
- **Spectral Peaks** - 频谱峰值检测（速度快）

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 使用桌面应用

```bash
python run_desktop.py
```

### 使用命令行

```bash
python -m audiomidi_app.cli --in input.wav --out output.mid
```

### 使用 Python API

```python
from audiomidi_app.transcribe import HarmonicSalienceTranscriber, SpectralPeaksTranscriber
from audiomidi_app.midi import events_to_midi
from audiomidi_app.audio import read_audio

# Read audio
audio = read_audio("input.wav", target_sr=None, mono=True)

# Choose engine
transcriber = HarmonicSalienceTranscriber()

# Transcribe
notes = transcriber.transcribe(audio.samples, audio.sample_rate)

# Save to MIDI
midi = events_to_midi(notes, bpm=120.0)
midi.save("output.mid")
```

## 项目结构

```
audiomidi_app/
├── audio.py      - 音频读取与处理
├── midi.py       - MIDI 文件生成
├── transcribe.py - 核心转录引擎
├── postprocess.py- 音符后处理
├── voice_separation.py - 声部分离
└── ui.py         - 桌面界面
```

## 测试系统

```bash
python quick_test.py
```

该脚本会生成一个 C 大和弦音频并测试转录功能。
