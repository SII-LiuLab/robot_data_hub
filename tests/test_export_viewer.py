import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pyarrow as pa
import pyarrow.parquet as pq
from scripts.viewer.export_viewer import FrameReader, read_dataset, read_episode, serve


def exported_dataset(tmp_path):
    root = tmp_path / 'export'
    episode = root / 'episodes' / 'sample'
    (episode / 'state').mkdir(parents=True)
    (episode / 'rgb').mkdir()
    record = {'episode_id': 'sample', 'cameras': ['top'],
              'instructions': [{'start_ns': 0, 'end_ns': 30, 'text': 'pick'}]}
    (root / 'episodes.jsonl').write_text(json.dumps(record) + '\n', encoding='utf-8')
    for name in ('left_eef', 'right_eef', 'left_gripper', 'right_gripper'):
        pose = name.endswith('_eef')
        values = [[0., 0., 0., 1., 0., 0., 0., 1., 0.]] * 2 if pose else [0., 1.]
        table = pa.table({'timestamp_ns': pa.array([0, 20], type=pa.int64()),
                          'pose' if pose else 'openness': pa.array(
                              values, type=pa.list_(pa.float64(), 9) if pose else pa.float32())})
        pq.write_table(table, episode / 'state' / f'{name}.parquet')
    pq.write_table(pa.table({'frame_index': pa.array([0, 1], type=pa.int64()),
                             'timestamp_ns': pa.array([10, 30], type=pa.int64())}),
                   episode / 'rgb' / 'top.parquet')
    (episode / 'rgb' / 'top.mp4').write_bytes(b'placeholder')
    return root, record


class ExportViewerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root, self.record = exported_dataset(Path(self.temporary.name))

    def test_reads_only_export_contract(self):
        episodes = read_dataset(self.root)
        self.assertEqual(episodes, {'sample': self.record})
        data = read_episode(self.root, self.record)
        self.assertEqual(data['end_ns'], 30)
        self.assertEqual(data['cameras']['top'], {'times': [10, 30], 'frames': 2})
        self.assertEqual(data['streams']['left_eef']['values'][0][3:9],
                         [1., 0., 0., 0., 1., 0.])

    def test_rejects_missing_or_unordered_camera_index(self):
        index = self.root / 'episodes' / 'sample' / 'rgb' / 'top.parquet'
        pq.write_table(pa.table({'frame_index': [0, 2], 'timestamp_ns': [10, 5]}), index)
        with self.assertRaisesRegex(ValueError, 'Invalid camera stream'):
            read_episode(self.root, self.record)

    def test_rejects_episode_path_escape(self):
        (self.root / 'episodes.jsonl').write_text(
            '{"episode_id":"../raw","cameras":["top"],"instructions":[]}\n',
            encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'episode_id'):
            read_dataset(self.root)

    def test_cancelled_browser_response_is_ignored(self):
        captured = {}

        class FakeServer:
            server_port = 8765

            def __init__(self, address, handler):
                captured['handler'] = handler

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def serve_forever(self):
                handler = captured['handler'].__new__(captured['handler'])
                handler.send_response = Mock()
                handler.send_header = Mock()
                handler.end_headers = Mock()
                handler.wfile = SimpleNamespace(write=Mock(side_effect=BrokenPipeError))
                handler.reply(200, b'image', 'image/jpeg')

        with patch('scripts.viewer.export_viewer.ThreadingHTTPServer', FakeServer):
            serve(self.root, '127.0.0.1', 8765)

    def test_frame_reader_uses_threaded_decode_and_requested_width(self):
        class FakeImage:
            def save(self, output, **_):
                output.write(b'jpeg')

        class FakeFrame:
            width = 1920
            height = 1200

            def __init__(self):
                self.reformat = Mock(return_value=SimpleNamespace(to_image=lambda: FakeImage()))

        stream = SimpleNamespace(thread_type=None)
        frame = FakeFrame()
        container = SimpleNamespace(
            streams=SimpleNamespace(video=[stream]),
            decode=Mock(return_value=iter([frame])),
            close=Mock(),
        )
        fake_av = SimpleNamespace(open=Mock(return_value=container))
        reader = FrameReader()
        with patch.dict(sys.modules, {'av': fake_av}):
            self.assertEqual(reader.get(Path('video.mp4'), 0, 320), b'jpeg')
        self.assertEqual(stream.thread_type, 'AUTO')
        frame.reformat.assert_called_once_with(width=320, height=200, format='rgb24')
        reader.close()

    def test_serves_local_3d_modules_without_exposing_other_files(self):
        results = {}

        class FakeServer:
            server_port = 8765

            def __init__(self, address, handler):
                self.handler = handler

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def serve_forever(self):
                for path in ('/', '/pose3d.js', '/vendor/three.module.min.js',
                             '/vendor/three.core.min.js', '/vendor/../../export_viewer.py'):
                    handler = self.handler.__new__(self.handler)
                    handler.path = path
                    handler.reply = Mock()
                    handler.do_GET()
                    results[path] = handler.reply.call_args.args

        with patch('scripts.viewer.export_viewer.ThreadingHTTPServer', FakeServer):
            serve(self.root, '127.0.0.1', 8765)
        self.assertIn(b'type="module"', results['/'][1])
        for path in ('/pose3d.js', '/vendor/three.module.min.js', '/vendor/three.core.min.js'):
            status, body, content_type = results[path]
            self.assertEqual(status, 200)
            self.assertTrue(body)
            self.assertTrue(content_type.startswith('text/javascript'))
        self.assertIn(b'three.core.min.js', results['/vendor/three.module.min.js'][1])
        self.assertEqual(results['/vendor/../../export_viewer.py'][0], 404)
