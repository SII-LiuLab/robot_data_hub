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

    def test_jump_steps_spikes_and_short_ramps_across_sample_rates(self):
        for hz in (30, 100, 300):
            times = np.arange(hz+1)/hz
            for shape in ('step', 'spike', 'ramp'):
                with self.subTest(hz=hz, shape=shape):
                    signal = (times >= .5).astype(float)
                    if shape == 'spike':
                        signal = (np.arange(len(times)) == hz//2).astype(float)
                    if shape == 'ramp':
                        signal = np.clip((times-.48)/.02, 0, 1)
                    streams = still(1)
                    streams['left_eef'] = stream(times, positions=.1*signal)
                    streams['right_eef'] = stream(times, angles=40*signal)
                    streams['left_gripper'] = stream(times, openness=.6*signal)
                    rows = detect(streams, 'state_jump')
                    self.assertEqual([r['stream'] for r in rows],
                                     ['left_eef', 'right_eef', 'left_gripper'])
                    for row in rows:
                        self.assertGreaterEqual(row['start_ns'], 450_000_000)
                        self.assertLessEqual(row['start_ns'], 510_000_000)
                        self.assertLessEqual(row['end_ns'], 600_000_000)

    def test_jump_constant_fast_motion_wrap_and_normal_start_stop(self):
        times = np.arange(301)/300
        for position in (10*times, 2*np.clip(times-.3, 0, .4), .5*np.sin(2*np.pi*times)):
            with self.subTest(position=position[100]):
                streams = still(1)
                streams['left_eef'] = stream(times, positions=position, angles=170+360*times)
                streams['right_gripper'] = stream(times, openness=np.clip(2*(times-.2), 0, 1))
                self.assertEqual(detect(streams, 'state_jump'), [])

    def test_jump_context_tolerates_sampling_just_below_30_hz(self):
        times = np.arange(31)*.034
        streams = still(2)
        for shape in ('spike', 'step'):
            signal = (np.arange(31) == 15) if shape == 'spike' else (np.arange(31) >= 15)
            streams['left_eef'] = stream(times, positions=.1*signal)
            with self.subTest(shape=shape):
                rows = detect(streams, 'state_jump')
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]['start_ns'], 510_000_000)

    def test_jump_small_noise_held_samples_and_irregular_times(self):
        rng = np.random.default_rng(42)
        times = np.r_[0, np.cumsum(rng.uniform(.002, .005, 300))]
        held = np.floor(times/.02)*.02
        streams = still(2)
        streams['left_eef'] = stream(times, positions=.2*held+rng.uniform(-.001, .001, len(times)),
                                     angles=20*held+rng.uniform(-.2, .2, len(times)))
        streams['left_gripper'] = stream(times, openness=.2+.2*held)
        self.assertEqual(detect(streams, 'state_jump'), [])
        streams['left_eef'].values[times >= .5, 0] += .1
        rows = detect(streams, 'state_jump')
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['start_ns'], streams['left_eef'].times[np.searchsorted(times, .5)])

    def test_jump_requires_amplitude_and_relative_speed_in_same_channel(self):
        times = np.arange(301)/300
        streams = still(1)
        # Fast continuous translation plus a tiny abrupt rotation is normal.
        streams['left_eef'] = stream(times, positions=10*times, angles=(times >= .5)*2)
        self.assertEqual(detect(streams, 'state_jump'), [])
        streams['left_eef'] = stream(times, positions=(times >= .5)*.02)
        self.assertEqual(detect(streams, 'state_jump'), [])
        config = replace(script.Config(), position_jump_min_m=.019)
        self.assertEqual(len(detect(streams, 'state_jump', config=config)), 1)

    def test_jump_duplicate_times_match_last_visible_sample(self):
        times = np.arange(301)/300
        duplicate = np.repeat(times, 2)
        positions = np.zeros(len(duplicate))
        positions[300] = 10  # Hidden by the last sample at the same timestamp.
        streams = still(1)
        streams['left_eef'] = stream(duplicate, positions=positions)
        self.assertEqual(detect(streams, 'state_jump'), [])
        positions[301] = .1
        streams['left_eef'] = stream(duplicate, positions=positions)
        self.assertEqual(len(detect(streams, 'state_jump')), 1)

    def test_jump_missing_context_and_recording_gaps_are_not_jumps(self):
        streams = still(1)
        streams['left_eef'] = stream([0, .01, .02], positions=[0, 10, 0])
        streams['right_eef'] = stream([0])
        self.assertEqual(detect(streams, 'state_jump'), [])
        times = np.r_[np.arange(101)/300, .7+np.arange(91)/300]
        streams['left_eef'] = stream(times, positions=(times >= .7)*10)
        self.assertEqual(detect(streams, 'state_jump'), [])
        times = np.arange(301)/300
        for jump_time in (.01, .99):
            streams['left_eef'] = stream(times, positions=(times >= jump_time)*.1)
            self.assertEqual(detect(streams, 'state_jump'), [])

    def test_jump_merges_channels_and_spike_edges_but_separates_events(self):
        times = np.arange(601)/300
        signal = ((times >= .5) & (times < .52) | (times >= 1.5)).astype(float)
        streams = still(2)
        streams['left_eef'] = stream(times, positions=.1*signal, angles=40*signal)
        rows = detect(streams, 'state_jump')
        self.assertEqual(len(rows), 2)
        self.assertEqual([r['start_ns'] for r in rows], [500_000_000, 1_500_000_000])
        self.assertGreater(rows[0]['end_ns'], 520_000_000)

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

    def test_internal_idle_does_not_mistake_smooth_motion_for_a_pause(self):
        # Large steps bracket smooth motion, so the old adjacent-frame rule
        # incorrectly reported the entire middle as idle.
        for hz in (30, 100, 300):
            times = np.arange(30*hz+1)/hz
            middle = np.clip(times-1, 0, 28)
            for name, channel, speed, step in (
                    ('left_eef', 'positions', .02, .1),
                    ('right_eef', 'angles', 10., 20.),
                    ('left_gripper', 'openness', .01, .1),
                    ('right_gripper', 'openness', .01, .1)):
                with self.subTest(hz=hz, channel=channel, name=name):
                    streams = still(30)
                    values = middle*speed + step*(times >= 1) + step*(times >= 29)
                    streams[name] = stream(times, **{channel: values})
                    self.assertEqual(detect(streams, 'internal_idle', end=30), [])

    def test_internal_idle_retains_real_pause_with_subthreshold_noise(self):
        times = np.arange(901)/30
        streams = still(30)
        signal = .1*(times >= 1) + .1*(times >= 20)
        streams['left_eef'] = stream(times, positions=signal+.001*np.sin(times*4))
        rows = detect(streams, 'internal_idle', end=30)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r['start_ns'] == 1_000_000_000 and r['end_ns'] == 20_000_000_000
                            for r in rows))

    def test_internal_idle_ignores_hidden_duplicate_samples(self):
        streams = still(30)
        times = np.repeat(np.arange(31), 2)
        positions = .1*(times >= 1) + .1*(times >= 20)
        positions[::2] += 10  # Viewer only displays the final value at each time.
        streams['left_eef'] = stream(times, positions=positions)
        rows = detect(streams, 'internal_idle', end=30)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r['start_ns'] == 1_000_000_000 and r['end_ns'] == 20_000_000_000
                            for r in rows))

    def test_config_rejects_invalid_thresholds(self):
        for kwargs in ({'max_boundary_idle_s': -1}, {'position_jump_min_m': float('nan')},
                       {'openness_tolerance': float('inf')}, {'jump_max_duration_s': 0},
                       {'jump_context_s': .01}, {'jump_speed_ratio': 1}):
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
