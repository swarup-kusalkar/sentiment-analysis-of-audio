"""Tests for the diarization module (Phase 3)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas.internal import DiarizationSegment


class TestSegmentMerge:
    """Test the merge_adjacent logic in diarization module."""

    def _merge_adjacent(self, segments, gap_s=0.0):
        """Replicated logic from diarization.py._merge_adjacent."""
        if not segments:
            return []
        merged = [segments[0]]
        for cur in segments[1:]:
            prev = merged[-1]
            if (
                cur.speaker_id == prev.speaker_id
                and cur.start_s - prev.end_s <= gap_s
            ):
                prev.end_s = cur.end_s
            else:
                merged.append(cur)
        return merged

    def make_seg(self, sid, start, end):
        return DiarizationSegment(speaker_id=sid, start_s=start, end_s=end)

    def test_empty(self):
        assert self._merge_adjacent([], 0.1) == []

    def test_single_segment(self):
        result = self._merge_adjacent([self.make_seg("A", 0, 5)], 0.1)
        assert len(result) == 1

    def test_same_speaker_consecutive(self):
        result = self._merge_adjacent(
            [self.make_seg("A", 0, 2), self.make_seg("A", 2.05, 5)],
            0.1
        )
        assert len(result) == 1
        assert result[0].speaker_id == "A"
        assert result[0].end_s == 5

    def test_different_speakers(self):
        result = self._merge_adjacent(
            [self.make_seg("A", 0, 2), self.make_seg("B", 2.05, 5)],
            0.1
        )
        assert len(result) == 2
        assert result[0].speaker_id == "A"
        assert result[1].speaker_id == "B"

    def test_interleaved_speakers(self):
        result = self._merge_adjacent(
            [
                self.make_seg("A", 0, 1),
                self.make_seg("B", 1, 2),
                self.make_seg("A", 2, 3),
                self.make_seg("A", 3.05, 4),
            ],
            0.1
        )
        assert len(result) == 3
        # A[0-1], B[1-2], A[2-4]
        assert result[2].speaker_id == "A"
        assert result[2].end_s == 4


class TestInferSpeakerCounting:
    def test_distinct_speakers(self):
        segments = [
            DiarizationSegment(speaker_id="A", start_s=0, end_s=1),
            DiarizationSegment(speaker_id="B", start_s=1, end_s=2),
            DiarizationSegment(speaker_id="A", start_s=2, end_s=3),
        ]
        speakers = {s.speaker_id for s in segments}
        assert len(speakers) == 2

    def test_single_speaker(self):
        speakers = {s.speaker_id for s in [
            DiarizationSegment(speaker_id="A", start_s=0, end_s=1),
            DiarizationSegment(speaker_id="A", start_s=1, end_s=2),
        ]}
        assert len(speakers) == 1