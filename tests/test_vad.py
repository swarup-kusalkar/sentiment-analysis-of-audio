"""Tests for the VAD module (Phase 2)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas.internal import VADResult, VADSegment


class TestMergeIntervals:
    """Test the interval merging logic used in VAD."""

    def _merge_intervals(self, segments, gap_s):
        """Replicate of _merge_intervals logic for testing."""
        if not segments:
            return []
        merged = [segments[0]]
        for cur in segments[1:]:
            prev = merged[-1]
            if cur.start_s - prev.end_s <= gap_s:
                prev.end_s = cur.end_s
            else:
                merged.append(cur)
        return merged

    def make_seg(self, start, end):
        return VADSegment(start_s=start, end_s=end)

    def test_empty(self):
        assert self._merge_intervals([], 0.4) == []

    def test_single_segment(self):
        result = self._merge_intervals([self.make_seg(1, 3)], 0.4)
        assert len(result) == 1
        assert result[0].start_s == 1

    def test_merge_disjoint(self):
        result = self._merge_intervals(
            [self.make_seg(0, 2), self.make_seg(3, 5)], 0.5
        )
        assert len(result) == 2
        assert result[0].end_s == 2
        assert result[1].end_s == 5

    def test_merge_adjacent(self):
        result = self._merge_intervals(
            [self.make_seg(0, 2), self.make_seg(2.2, 5.1)],
            0.5
        )
        assert len(result) == 1
        assert result[0].start_s == 0
        assert result[0].end_s == 5.1

    def test_merge_chain(self):
        result = self._merge_intervals(
            [
                self.make_seg(0, 2),
                self.make_seg(2.1, 4),
                self.make_seg(4.1, 6),
            ],
            0.4
        )
        assert len(result) == 1
        assert result[0].start_s == 0
        assert result[0].end_s == 6


class TestVADResult:
    def test_vad_result_speeds(self):
        # Verify the schema we defined
        result = VADResult(has_speech=True, segments=[
            VADSegment(start_s=0.0, end_s=1.5),
            VADSegment(start_s=2.0, end_s=3.5),
        ], speech_duration_s=3.0)
        assert result.has_speech is True
        assert len(result.segments) == 2
        assert result.speech_duration_s == 3.0

    def test_no_speech_empty(self):
        result = VADResult(has_speech=False, segments=[], speech_duration_s=0.0)
        assert result.has_speech is False
        assert result.segments == []
        assert result.speech_duration_s == 0.0