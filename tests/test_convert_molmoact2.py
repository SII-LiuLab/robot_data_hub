import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.convert import molmoact2 as script
from scripts.robot.kinematics import fk_poses
from scripts.robot.urdf_model import parse_urdf

MODEL = script.ROOT / 'assets/robot_models/yam'


def make_source(root, frames=4):
    meta = root / 'source/meta'
    meta.mkdir(parents=True)
    (meta / 'info.json').write_text(json.dumps({
        'robot_type': 'bi_yam_follower', 'codebase_version': 'v3.0',
        'features': {'observation.state': {'shape': [14], 'names': script.STATE_NAMES},
                     **{key: {'dtype': 'video'} for key in script.CAMERAS}},
    }))
    pq.write_table(pa.table({'task_index': [0, 3], '__index_level_0__': ['拿起杯子', 'Put down']}),
                   meta / 'tasks.parquet')
    pq.write_table(pa.table({'episode_index': [7], 'task': ['整理杯子']}),
                   meta / 'tasks_annotated.parquet')
    episode = root / 'episodes/episode_000007'
    (episode / 'videos').mkdir(parents=True)
    pq.write_table(pa.table({
        'observation.state': pa.array([
            [.2, -.3, .5, -.6, .4, .1, a, -.1, .4, -.2, .5, -.3, .6, b]
            for a, b in zip([-.1, .25, 1.2, 1], [1, .75, .5, 0])],
            type=pa.list_(pa.float32(), 14)),
        # Nonzero origin, irregular intervals; container FPS must not determine time.
        'timestamp': pa.array([2., 2.031, 2.079, 2.141], type=pa.float32()),
        'frame_index': [0, 1, 2, 3], 'episode_index': [7] * 4,
        'index': [100, 101, 102, 103], 'task_index': [0, 0, 3, 3],
        'action': [[float('nan')] * 14] * 4,
    }), episode / 'data.parquet')
    (episode / 'episode.json').write_text(json.dumps({
        'episode_index': 7, 'length': 4, 'dataset_from_index': 100, 'dataset_to_index': 104,
        'tasks': ['拿起杯子', 'Put down'], 'task': 'stale cached annotation',
        'videos': {key: {'from_timestamp': i * 100, 'to_timestamp': i * 100 + 1}
                   for i, key in enumerate(script.CAMERAS)},
    }))
    for key in script.CAMERAS:
        with av.open(str(episode / 'videos' / f'{key}.mp4'), 'w') as output:
            stream = output.add_stream('libx264', rate=25)
            stream.width = stream.height = 32
            stream.pix_fmt = 'yuv420p'
            for i in range(frames):
                frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i * 60, np.uint8), format='rgb24')
                frame.pts = i
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
    return episode


class MolmoAct2Tests(unittest.TestCase):
    @unittest.skipUnless(MODEL.exists(), 'YAM asset package not installed')
    def test_full_export_schema_state_time_annotations_and_video(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = make_source(root / 'raw')
            script.convert(root / 'raw', root / 'out')
            records = [json.loads(line) for line in (root / 'out/episodes.jsonl').read_text().splitlines()]
            self.assertEqual(len(records), 1)
            record = records[0]
            exported = root / 'out/episodes' / episode.name
            source = pq.read_table(episode / 'data.parquet').to_pydict()
            times = [script.seconds_to_ns(t) - 2_000_000_000 for t in source['timestamp']]
            self.assertEqual(record, {'episode_id': episode.name, 'cameras': ['left_wrist', 'right_wrist', 'top'],
                                     'instructions': [{'start_ns': 0, 'end_ns': times[-1], 'text': '整理杯子'}]})
            model = parse_urdf(MODEL / 'robot.urdf')
            for side, offset, grips in [('left', 0, [0, .25, 1, 1]), ('right', 7, [1, .75, .5, 0])]:
                poses = pq.read_table(exported / 'state' / f'{side}_eef.parquet')
                self.assertEqual(poses['timestamp_ns'].to_pylist(), times)
                self.assertEqual(poses.schema.field('pose').type, pa.list_(pa.float64(), 9))
                self.assertFalse(poses.schema.field('pose').nullable)
                # Independent expected FK: remove world mounting and apply TCP/axes at wrist.
                q = source['observation.state'][0][offset:offset + 6]
                transforms = fk_poses(model, {f'arm_{side}_joint{i+1}': value for i, value in enumerate(q)})
                wrist = np.linalg.inv(transforms[f'arm_{side}_base_link']) @ transforms[f'arm_{side}_link6']
                expected = np.r_[wrist[:3, 3] + .1347 * wrist[:3, 2], wrist[:3, 2], wrist[:3, 0]]
                np.testing.assert_allclose(poses['pose'][0].as_py(), expected, atol=1e-9)
                gripper = pq.read_table(exported / 'state' / f'{side}_gripper.parquet')
                self.assertEqual(gripper['timestamp_ns'].to_pylist(), times)
                self.assertEqual(gripper['openness'].type, pa.float32())
                self.assertEqual(gripper['openness'].to_pylist(), grips)
            for key, camera in script.CAMERAS.items():
                video = exported / 'rgb' / f'{camera}.mp4'
                self.assertEqual(video.read_bytes(), (episode / 'videos' / f'{key}.mp4').read_bytes())
                table = pq.read_table(exported / 'rgb' / f'{camera}.parquet')
                self.assertEqual(table['timestamp_ns'].to_pylist(), times)
                self.assertEqual(table['frame_index'].to_pylist(), list(range(4)))
                with av.open(str(video)) as reader:
                    self.assertEqual(reader.streams.video[0].codec_context.name, 'h264')
                    self.assertEqual(len(reader.streams.audio), 0)
                    for i, frame in enumerate(reader.decode(video=0)):
                        self.assertAlmostEqual(frame.to_ndarray(format='rgb24').mean(), i * 60, delta=3)
            with self.assertRaisesRegex(ValueError, 'already exists'):
                script.convert(root / 'raw', root / 'out')

    def test_annotation_fallback_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            episode = make_source(root)
            meta = root / 'source/meta'
            for annotation in (None, '', '  '):
                pq.write_table(pa.table({'episode_index': [7], 'task': [annotation]}), meta / 'tasks_annotated.parquet')
                tasks, annotated = script.load_tasks(meta)
                with patch.object(script, 'YamFK') as fk:
                    times, _, instruction = script.read_episode(episode, tasks, annotated, fk)
                self.assertEqual(instruction, [
                    {'start_ns': 0, 'end_ns': times[2] - times[0], 'text': '拿起杯子'},
                    {'start_ns': times[2] - times[0], 'end_ns': times[-1] - times[0], 'text': 'Put down'}])
            (meta / 'tasks_annotated.parquet').unlink()
            self.assertEqual(script.load_tasks(meta)[1], {})
            pq.write_table(pa.table({'episode_index': [7, 7], 'task': ['A', 'B']}), meta / 'tasks_annotated.parquet')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                script.load_tasks(meta)

    def test_bad_rows_fail_before_copying_video(self):
        mutations = [
            ('timestamp', [2., 1., 3., 4.]), ('timestamp', [2.] * 4),
            ('timestamp', [None, 2., 3., 4.]), ('timestamp', [0., float('inf'), 2., 3.]),
            ('frame_index', [0, 2, 1, 3]), ('index', [100, 101, 101, 103]),
            ('episode_index', [7, 7, 8, 7]), ('task_index', [0, 0, 99, 3]),
            ('observation.state', [[0.] * 13] * 4),
            ('observation.state', [[0.] * 6 + [float('nan')] + [0.] * 7] * 4),
            ('observation.state', [[float('inf')] + [0.] * 13] * 4),
        ]
        for key, value in mutations:
            with self.subTest(key=key, value=value), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                episode = make_source(root / 'raw')
                rows = pq.read_table(episode / 'data.parquet').to_pydict()
                rows[key] = value
                pq.write_table(pa.table(rows), episode / 'data.parquet')
                with patch.object(script, 'YamFK'), patch.object(script, 'copy_h264_video') as video:
                    with self.assertRaises(ValueError):
                        script.convert(root / 'raw', root / 'out')
                    video.assert_not_called()
                self.assertFalse((root / 'out').exists())
                self.assertFalse(list(root.glob('.out-*')))

    @unittest.skipUnless(MODEL.exists(), 'YAM asset package not installed')
    def test_failed_video_or_missing_episode_does_not_publish(self):
        for missing_episode in (False, True):
            with self.subTest(missing_episode=missing_episode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                make_source(root / 'raw', frames=4 if missing_episode else 3)
                if missing_episode:
                    (root / 'raw/episodes/episode_000008').mkdir()
                with self.assertRaises((ValueError, FileNotFoundError)):
                    script.convert(root / 'raw', root / 'out')
                self.assertFalse((root / 'out').exists())
                self.assertFalse(list(root.glob('.out-*')))

    def test_wrong_layout_and_invalid_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_source(root / 'raw')
            path = root / 'raw/source/meta/info.json'
            info = json.loads(path.read_text())
            script.validate_source(info)
            info['features']['observation.state']['names'].reverse()
            with self.assertRaisesRegex(ValueError, 'layout'):
                script.validate_source(info)
            for limit in (0, -1, True):
                with self.assertRaisesRegex(ValueError, 'limit'):
                    script.convert(root / 'raw', root / 'out', limit=limit)


if __name__ == '__main__':
    unittest.main()
