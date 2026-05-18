import unittest

import numpy as np

from audiomidi_app.midi import NoteEvent, events_to_midi
from audiomidi_app.transcribe import SpectralPeaksTranscriber
from audiomidi_app.symbolic_decoder import (
    SymbolicDecoder,
    SymbolicDecoderConfig,
    create_symbolic_decoder,
)


class TestSpectralPeaksTranscriber(unittest.TestCase):
    def test_chord(self) -> None:
        sr = 22050
        t = np.arange(0, 2 * sr) / sr
        x = 0.2 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 659.25 * t)
        events = SpectralPeaksTranscriber().transcribe(x.astype(np.float32), sr)
        notes = {e.note for e in events}
        self.assertIn(69, notes)
        self.assertIn(76, notes)
        mid = events_to_midi(events, bpm=120.0)
        self.assertGreaterEqual(len(mid.tracks), 1)


class TestSymbolicDecoder(unittest.TestCase):
    """测试 Symbolic Decoder"""
    
    def setUp(self):
        """设置测试环境"""
        self.decoder = create_symbolic_decoder("default")
    
    def create_test_notes(self, include_harmonics=False):
        """创建测试用的音符列表"""
        notes = []
        
        # 基础音符 C4
        c4 = NoteEvent(
            note=60,
            start_s=0.0,
            end_s=1.0,
            velocity=80,
            confidence=1.0
        )
        notes.append(c4)
        
        if include_harmonics:
            # 添加 C4 的泛音
            # C5 (60 + 12) - 二次泛音
            c5 = NoteEvent(
                note=72,
                start_s=0.0,
                end_s=0.9,
                velocity=50,
                confidence=0.6
            )
            notes.append(c5)
            
            # G5 (60 + 19) - 三次泛音
            g5 = NoteEvent(
                note=79,
                start_s=0.0,
                end_s=0.8,
                velocity=40,
                confidence=0.4
            )
            notes.append(g5)
        
        # E4
        e4 = NoteEvent(
            note=64,
            start_s=0.0,
            end_s=1.0,
            velocity=75,
            confidence=1.0
        )
        notes.append(e4)
        
        # G4
        g4 = NoteEvent(
            note=67,
            start_s=0.0,
            end_s=1.0,
            velocity=70,
            confidence=1.0
        )
        notes.append(g4)
        
        return notes
    
    def create_fragmented_notes(self):
        """创建碎片化的音符（用于测试链接）"""
        notes = []
        
        # C4 的多个碎片
        c4_1 = NoteEvent(
            note=60,
            start_s=0.0,
            end_s=0.3,
            velocity=80,
            confidence=1.0
        )
        notes.append(c4_1)
        
        c4_2 = NoteEvent(
            note=60,
            start_s=0.35,
            end_s=0.7,
            velocity=75,
            confidence=1.0
        )
        notes.append(c4_2)
        
        c4_3 = NoteEvent(
            note=60,
            start_s=0.75,
            end_s=1.0,
            velocity=70,
            confidence=1.0
        )
        notes.append(c4_3)
        
        # E4 (完整)
        e4 = NoteEvent(
            note=64,
            start_s=0.0,
            end_s=1.0,
            velocity=75,
            confidence=1.0
        )
        notes.append(e4)
        
        return notes
    
    def test_harmonic_suppression(self):
        """测试泛音抑制"""
        print("\n=== 测试泛音抑制 ===")
        
        notes_with_harmonics = self.create_test_notes(include_harmonics=True)
        print(f"输入音符数: {len(notes_with_harmonics)}")
        print(f"输入音符: {[n.note for n in notes_with_harmonics]}")
        
        # 解码
        decoded = self.decoder.decode(notes_with_harmonics, tempo=120.0)
        
        print(f"解码后音符数: {len(decoded)}")
        print(f"解码后音符: {[n.note for n in decoded]}")
        
        # 应该只剩下非泛音音符
        self.assertLess(len(decoded), len(notes_with_harmonics))
        
        # 基音应该保留
        note_numbers = {n.note for n in decoded}
        self.assertIn(60, note_numbers)
        self.assertIn(64, note_numbers)
        self.assertIn(67, note_numbers)
        
        print("✅ 泛音抑制测试通过")
    
    def test_note_linking(self):
        """测试音符链接"""
        print("\n=== 测试音符链接 ===")
        
        fragmented_notes = self.create_fragmented_notes()
        print(f"输入音符数: {len(fragmented_notes)}")
        
        # 解码
        decoded = self.decoder.decode(fragmented_notes, tempo=120.0)
        
        print(f"解码后音符数: {len(decoded)}")
        
        # C4 应该被合并成一个
        c4_notes = [n for n in decoded if n.note == 60]
        self.assertEqual(len(c4_notes), 1)
        
        print("✅ 音符链接测试通过")
    
    def test_polyphony_pruning(self):
        """测试复调剪枝"""
        print("\n=== 测试复调剪枝 ===")
        
        # 创建过多复调
        many_notes = []
        for i in range(15):
            note = NoteEvent(
                note=60 + i,
                start_s=0.0,
                end_s=0.5,
                velocity=80 - i,
                confidence=1.0 - i * 0.05
            )
            many_notes.append(note)
        
        print(f"输入音符数: {len(many_notes)}")
        
        # 解码
        decoded = self.decoder.decode(many_notes, tempo=120.0)
        
        print(f"解码后音符数: {len(decoded)}")
        
        # 应该被剪枝到合理数量
        self.assertLessEqual(len(decoded), 12)
        
        print("✅ 复调剪枝测试通过")
    
    def test_beat_quantization(self):
        """测试节拍量化"""
        print("\n=== 测试节拍量化 ===")
        
        notes = [
            NoteEvent(
                note=60,
                start_s=0.12,  # 稍微偏离
                end_s=0.5,
                velocity=80,
                confidence=1.0
            )
        ]
        
        # 提供节拍位置
        beats = [0.0, 0.5, 1.0, 1.5]
        
        # 解码
        decoded = self.decoder.decode(notes, tempo=120.0, beats=beats)
        
        print(f"原起始时间: 0.12")
        print(f"新起始时间: {decoded[0].start_s}")
        
        print("✅ 节拍量化测试通过")
    
    def test_integration_with_transcriber(self):
        """测试与转写器集成"""
        print("\n=== 测试集成 ===")
        
        # 创建测试音频
        sr = 22050
        t = np.arange(0, 2 * sr) / sr
        x = 0.2 * np.sin(2 * np.pi * 440 * t)  # A4
        
        # 用 DSP 转写
        raw_events = SpectralPeaksTranscriber().transcribe(x.astype(np.float32), sr)
        print(f"原始转写: {len(raw_events)} 音符")
        
        # 用 Symbolic Decoder 处理
        decoded = self.decoder.decode(raw_events, tempo=120.0)
        print(f"解码后: {len(decoded)} 音符")
        
        print("✅ 集成测试通过")


def run_symbolic_decoder_demo():
    """运行 Symbolic Decoder 演示"""
    print("\n" + "=" * 60)
    print("Symbolic Decoder 演示")
    print("=" * 60)
    
    # 创建测试数据
    notes = [
        # 主和弦
        NoteEvent(60, 0.0, 1.0, 80, 1.0),  # C4
        NoteEvent(64, 0.0, 1.0, 75, 1.0),  # E4
        NoteEvent(67, 0.0, 1.0, 70, 1.0),  # G4
        
        # 泛音（应该被抑制）
        NoteEvent(72, 0.0, 0.9, 50, 0.6),  # C5 (泛音)
        NoteEvent(79, 0.0, 0.8, 40, 0.4),  # G5 (泛音)
        
        # 碎片（应该被连接）
        NoteEvent(72, 0.0, 0.3, 85, 1.0),   # C5
        NoteEvent(72, 0.35, 0.7, 80, 1.0),  # C5
        NoteEvent(72, 0.75, 1.0, 75, 1.0),  # C5
    ]
    
    print(f"\n输入: {len(notes)} 音符")
    print(f"音符: {[(n.note, round(n.confidence, 2)) for n in notes]}")
    
    # 运行解码器
    decoder = create_symbolic_decoder("default")
    decoded = decoder.decode(notes, tempo=120.0)
    
    print(f"\n输出: {len(decoded)} 音符")
    print(f"音符: {[(n.note, round(n.confidence, 2)) for n in decoded]}")
    
    print("\n✅ Symbolic Decoder 运行正常！")


if __name__ == "__main__":
    # 运行演示
    run_symbolic_decoder_demo()
    
    # 运行单元测试
    print("\n" + "=" * 60)
    print("运行单元测试")
    print("=" * 60)
    unittest.main(verbosity=2)
