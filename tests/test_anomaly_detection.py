from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.check import anomaly_detection as script
from scripts.convert.export_common import write_state, write_video_index


def stream(seconds, positions=None, angles=None, openness=None):
    times = np.rint(np.asarray(seconds) * 1e9).astype(np.int64)
    if openness is not None:
        return script.StateStream(times, np.asarray(openness, dtype=float))
    n = len(times)
    angles = np.deg2rad(np.zeros(n) if angles is None else angles)
    values = np.zeros((n, 9))
    if positions is not None:
        p = np.asarray(positions)
        if p.ndim == 1:
            values[:, 0] = p
        else:
            values[:, :3] = p
    values[:, 3] = np.cos(angles)
    values[:, 4] = np.sin(angles)
    values[:, 6] = -np.sin(angles)
    values[:, 7] = np.cos(angles)
    return script.StateStream(times, values, script.rotation_quaternions(values))


def still(end=20):
    return {name: stream([0, end], **({'openness': [0, 0]} if 'gripper' in name else {}))
            for name in script.STATE_NAMES}


def detect(streams, reason, end=20, config=script.Config()):
    return [row for row in script.detect_episode(3, streams, int(end*1e9), config)
            if row['reason'] == reason]


class DetectionTests(unittest.TestCase):
    def test_boundary_uses_minimum_across_streams_and_final_reference(self):
        streams = still()
        streams['left_eef'] = stream([1, 4, 5, 14, 15, 18], positions=[0, 0, .1, .2, .3, .3])
        streams['right_gripper'] = stream([2, 7, 16, 19], openness=[0, 1, 0, 0])
        self.assertEqual(detect(streams, 'boundary_idle'), [
            dict(index=3, stream='left_eef', start_ns=0, end_ns=5_000_000_000, reason='boundary_idle'),
            dict(index=3, stream='left_eef', start_ns=14_000_000_000, end_ns=20_000_000_000,
                 reason='boundary_idle')])

    def test_boundary_all_static_two_records_and_ties_are_stable(self):
        rows = detect(still(), 'boundary_idle')
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row['stream'] == 'left_eef' and row['start_ns'] == 0
                            and row['end_ns'] == 20_000_000_000 for row in rows))
        self.assertEqual(detect(still(), 'internal_idle'), [])
        self.assertEqual(detect(still(3), 'boundary_idle', end=3), [])

    def test_boundary_orientation_openness_and_overrides(self):
        streams = still()
        streams['right_eef'] = stream([0, 1, 19, 20], angles=[0, 6, 12, 18])
        self.assertEqual(detect(streams, 'boundary_idle'), [])
        config = replace(script.Config(), orientation_tolerance_deg=20)
        self.assertEqual(len(detect(streams, 'boundary_idle', config=config)), 2)
        streams['left_gripper'] = stream([0, 1, 19, 20], openness=[0, .1, .2, .3])
        self.assertEqual(detect(streams, 'boundary_idle', config=config), [])

    def test_jump_nonuniform_timing_and_thresholds(self):
        streams = still(1)
        # Residual .01 m; dt .01 and .02 s => exactly 100 m/s².
        streams['left_eef'] = stream([0, .01, .03], positions=[0, .01, 0])
        self.assertEqual(detect(streams, 'state_jump'), [])
        rows = detect(streams, 'state_jump', config=replace(script.Config(), position_acceleration_limit=99))
        self.assertEqual([(r['stream'], r['start_ns'], r['end_ns']) for r in rows],
                         [('left_eef', 10_000_000, 10_000_000)])
        streams['right_gripper'] = stream([0, .01, .03], openness=[0, .1, 0])
        self.assertEqual([r['stream'] for r in detect(streams, 'state_jump')], ['right_gripper'])

    def test_jump_fast_constant_velocity_and_slerp_across_wrap(self):
        streams = still(1)
        streams['left_eef'] = stream([0, .01, .03], positions=[0, 10, 30], angles=[170, 180, 200])
        streams['right_gripper'] = stream([0, .01, .03], openness=[0, .2, .6])
        self.assertEqual(detect(streams, 'state_jump'), [])
        streams['left_eef'] = stream([0, .01, .03], angles=[170, 175, 200])
        self.assertEqual(len(detect(streams, 'state_jump')), 1)

    def test_jump_duplicate_timestamps_and_short_streams(self):
        streams = still(1)
        streams['left_eef'] = stream([0, .01, .01, .02], positions=[0, 10, 20, 0])
        streams['right_eef'] = stream([0])
        self.assertEqual(detect(streams, 'state_jump'), [])

    def test_rotations_near_half_turn_on_each_axis(self):
        poses = []
        for rotation in (np.eye(3), np.diag([1, -1, -1]),
                         np.diag([-1, 1, -1]), np.diag([-1, -1, 1])):
            poses.append(np.r_[np.zeros(3), rotation[:, 0], rotation[:, 1]])
        quaternions = script.rotation_quaternions(np.array(poses))
        np.testing.assert_allclose(quaternions, np.eye(4))
        np.testing.assert_allclose(script.rotation_distance_deg(quaternions[0], quaternions[1:]), 180)
        np.testing.assert_allclose(script.rotation_distance_deg(quaternions, -quaternions), 0)

    def test_internal_idle_merges_native_changes_and_emits_four_records(self):
        streams = still(30)
        streams['left_eef'] = stream([0, 1, 5, 15, 30], positions=[0, .1, .1, .2, .2])
        streams['right_gripper'] = stream([0, 2, 15, 29], openness=[0, .2, .4, .4])
        rows = detect(streams, 'internal_idle', end=30)
        self.assertEqual([r['stream'] for r in rows], list(script.STATE_NAMES))
        self.assertTrue(all(r['start_ns'] == 2_000_000_000 and r['end_ns'] == 15_000_000_000
                            for r in rows))
        self.assertEqual(detect(streams, 'internal_idle', end=30, config=replace(
            script.Config(), max_internal_idle_s=13)), [])
        streams['left_gripper'] = stream(np.arange(31), openness=np.arange(31) % 2)
        self.assertEqual(detect(streams, 'internal_idle', end=30), [])

    def test_internal_changes_compare_adjacent_samples_not_boundary(self):
        streams = still()
        streams['left_eef'] = stream(np.arange(20), positions=np.arange(20)*.009)
        self.assertEqual(detect(streams, 'internal_idle'), [])

    def test_config_rejects_invalid_thresholds(self):
        for kwargs in ({'max_boundary_idle_s': -1}, {'position_acceleration_limit': float('nan')},
                       {'openness_tolerance': float('inf')}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                script.Config(**kwargs)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / 'episodes/sample'
        for name, data in still(2).items():
            write_state(self.base / 'state' / f'{name}.parquet', data.times, data.values, 0,
                        pose=name.endswith('_eef'))
        (self.base / 'rgb').mkdir()
        write_video_index(self.base / 'rgb/top.parquet', [0, 5_000_000_000], 0)
        self.record = {'episode_id': 'sample', 'cameras': ['top'],
                       'instructions': [{'start_ns': 0, 'end_ns': 99_000_000_000, 'text': 'test'}]}
        self.manifest = self.root / 'episodes.jsonl'
        self.manifest.write_text('\n' + json.dumps(self.record) + '\n', encoding='utf-8')

    def test_jsonl_physical_line_index_camera_endpoint_and_no_video_decode(self):
        self.assertEqual(script.detect_dataset(self.root), (1, 2))
        rows = [json.loads(line) for line in (self.root / 'anomalies.jsonl').read_text().splitlines()]
        self.assertTrue(all(row == dict(index=1, stream='left_eef', start_ns=0,
                                       end_ns=5_000_000_000, reason='boundary_idle') for row in rows))

    def test_failure_preserves_previous_report_and_removes_temporary_file(self):
        output = self.root / 'report.jsonl'
        output.write_text('previous\n')
        self.manifest.write_text(json.dumps(self.record) + '\n{}\n')
        with self.assertRaisesRegex(ValueError, 'line 2'):
            script.detect_dataset(self.root, output)
        self.assertEqual(output.read_text(), 'previous\n')
        self.assertEqual(list(self.root.glob('.*.tmp')), [])

    def test_rejects_invalid_state_and_protects_inputs(self):
        for times, values in (([2, 1], [0, 0]), ([0, 1], [float('nan'), 0]),
                              ([0, 1], [0, 2]), ([0, None], [0, 0])):
            pq.write_table(pa.table({'timestamp_ns': pa.array(times, type=pa.int64()),
                                     'openness': values}), self.base / 'state/left_gripper.parquet')
            with self.subTest(times=times, values=values), self.assertRaises(ValueError):
                script.detect_dataset(self.root)
        for output in (self.manifest, self.base / 'report.jsonl', self.root / 'state.parquet'):
            with self.assertRaises(ValueError):
                script.detect_dataset(self.root, output)

    def test_cli_both_entrypoints_and_threshold_override(self):
        repo = Path(__file__).resolve().parents[1]
        for entry in (['scripts/check/anomaly_detection.py'], ['-m', 'scripts.check.anomaly_detection']):
            result = subprocess.run([sys.executable, *entry, str(self.root), '--max-boundary-idle-s', '5'],
                                    cwd=repo, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('wrote 0 anomalies', result.stdout)
            self.assertEqual((self.root / 'anomalies.jsonl').read_text(), '')


if __name__ == '__main__':
    unittest.main()
