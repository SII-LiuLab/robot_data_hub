"""Local viewer for the exported dataset contract."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from fractions import Fraction
from pathlib import Path
import re
from threading import Lock
from urllib.parse import parse_qs, urlsplit

import pyarrow.parquet as pq


HERE = Path(__file__).resolve().parent
STATE_NAMES = ('left_eef', 'right_eef', 'left_gripper', 'right_gripper')
CAMERA_ID = re.compile(r'^[a-z0-9_]+$')


def manifest_records(root):
    # The anomaly contract uses physical line numbers, including blank lines.
    with (root / 'episodes.jsonl').open(encoding='utf-8') as source:
        for index, line in enumerate(source):
            if line.strip():
                yield index, json.loads(line)


def read_dataset(root):
    records = [record for _, record in manifest_records(root)]
    if not records:
        raise ValueError('episodes.jsonl is empty')
    episodes = {}
    for record in records:
        episode_id = record['episode_id']
        # A manifest entry must address one direct child of episodes/.
        if (not isinstance(episode_id, str) or episode_id in ('', '.', '..')
                or '/' in episode_id or '\\' in episode_id or episode_id in episodes):
            raise ValueError(f'Invalid or duplicate episode_id: {episode_id!r}')
        cameras = record['cameras']
        if not cameras or len(cameras) != len(set(cameras)) or any(
                not isinstance(c, str) or not CAMERA_ID.fullmatch(c) for c in cameras):
            raise ValueError(f'Invalid cameras in episode {episode_id}')
        episodes[episode_id] = record
    return episodes


def read_anomalies(root, path):
    """Resolve the anomaly record contract without importing detection rules."""
    episode_ids = {index: record['episode_id'] for index, record in manifest_records(root)}
    records = []
    with path.open(encoding='utf-8') as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError('Expected an anomaly object')
                for field in ('index', 'start_ns', 'end_ns'):
                    if type(record.get(field)) is not int or record[field] < 0:
                        raise ValueError(f'{field} must be a nonnegative integer')
                if record['index'] not in episode_ids:
                    raise ValueError('index does not address an episode manifest record')
                if record.get('stream') not in STATE_NAMES:
                    raise ValueError('Invalid stream')
                if record['end_ns'] < record['start_ns']:
                    raise ValueError('end_ns must be >= start_ns')
                if not isinstance(record.get('reason'), str) or not record['reason'].strip():
                    raise ValueError('reason must be a nonempty string')
                records.append({key: record[key] for key in
                                ('index', 'stream', 'start_ns', 'end_ns', 'reason')})
                records[-1]['episode_id'] = episode_ids[record['index']]
            except ValueError as exc:
                raise ValueError(f'{path}:{line_number}: {exc}') from exc
    return records


def read_episode(root, record):
    base = root / 'episodes' / record['episode_id']
    streams = {}
    for name in STATE_NAMES:
        value_name = 'pose' if name.endswith('_eef') else 'openness'
        table = pq.read_table(base / 'state' / f'{name}.parquet',
                              columns=['timestamp_ns', value_name])
        times = table['timestamp_ns'].to_pylist()
        values = table[value_name].to_pylist()
        if not times or any(b < a for a, b in zip(times, times[1:])):
            raise ValueError(f'Empty or unordered state stream: {name}')
        streams[name] = {'times': times, 'values': values}
    cameras = {}
    for camera in record['cameras']:
        table = pq.read_table(base / 'rgb' / f'{camera}.parquet',
                              columns=['frame_index', 'timestamp_ns'])
        frames = table['frame_index'].to_pylist()
        times = table['timestamp_ns'].to_pylist()
        if (not times or frames != list(range(len(frames)))
                or any(b < a for a, b in zip(times, times[1:]))
                or not (base / 'rgb' / f'{camera}.mp4').is_file()):
            raise ValueError(f'Invalid camera stream: {camera}')
        cameras[camera] = {'times': times, 'frames': len(frames)}
    end_ns = max([stream['times'][-1] for stream in streams.values()] +
                 [camera['times'][-1] for camera in cameras.values()])
    return {'episode_id': record['episode_id'], 'instructions': record['instructions'],
            'streams': streams, 'cameras': cameras, 'end_ns': end_ns}


class FrameReader:
    """Decode by display-order frame index; MP4 timestamps are never acquisition times."""

    def __init__(self):
        self.lock = Lock()
        self.path = None
        self.container = None
        self.iterator = None
        self.index = -1
        self.last_frame = None
        self.last_images = {}
        self.seeked = False

    def close(self):
        if self.container is not None:
            self.container.close()
        self.path = self.container = self.iterator = self.last_frame = None
        self.last_images = {}
        self.index = -1
        self.seeked = False

    def _open(self, path):
        self.close()
        import av
        self.container = av.open(str(path))
        self.container.streams.video[0].thread_type = 'AUTO'
        self.iterator = self.container.decode(video=0)
        self.path = path

    def _seek(self, index):
        stream = self.container.streams.video[0]
        # Exported MP4s have one display frame per tick of their 30 Hz clock.
        # Other videos retain the sequential, display-order fallback.
        if (getattr(stream, 'average_rate', None) != 30
                or not getattr(stream, 'time_base', None)
                or not hasattr(self.container, 'seek')):
            return False
        origin = stream.start_time or 0
        target = origin + int(Fraction(index, 30) / stream.time_base)
        self.container.seek(target, stream=stream, backward=True)
        self.iterator = self.container.decode(video=0)
        self.index = -1
        self.seeked = True
        return True

    def get(self, path, index, width):
        with self.lock:
            return self._get(path, index, width)

    def _get(self, path, index, width):
        if self.path != path:
            self._open(path)
        if index < self.index or index - self.index > 120:
            if not self._seek(index):
                if index < self.index:
                    self._open(path)
        while self.index < index:
            frame = next(self.iterator)
            if self.seeked:
                stream = self.container.streams.video[0]
                if frame.pts is None or frame.time_base is None:
                    self._open(path)
                    continue
                ordinal = round((frame.pts - (stream.start_time or 0)) * frame.time_base * 30)
                if ordinal < 0 or (self.index >= 0 and ordinal != self.index + 1) or ordinal > index:
                    self._open(path)
                    continue
                self.index = ordinal
            else:
                self.index += 1
            if self.index == index:
                self.last_frame = frame
                self.last_images = {}
                self.seeked = False
        if width not in self.last_images:
            frame = self.last_frame
            if frame.width > width:
                height = max(1, round(frame.height * width / frame.width))
                frame = frame.reformat(width=width, height=height, format='rgb24')
            image = frame.to_image()
            output = BytesIO()
            image.save(output, format='JPEG', quality=80)
            self.last_images[width] = output.getvalue()
        return self.last_images[width]


def serve(root, host, port, anomalies_path=None):
    episodes = read_dataset(root)
    if anomalies_path is None and (root / 'anomalies.jsonl').is_file():
        anomalies_path = root / 'anomalies.jsonl'
    anomalies = read_anomalies(root, anomalies_path) if anomalies_path is not None else None
    readers = {}
    readers_lock = Lock()
    static_files = {
        '/': ((HERE / 'index.html').read_bytes(), 'text/html; charset=utf-8'),
        **{'/' + name: ((HERE / name).read_bytes(), 'text/javascript; charset=utf-8')
           for name in ('pose3d.js', 'anomalies.js', 'vendor/three.module.min.js',
                        'vendor/three.core.min.js')},
    }

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def reply(self, status, body, content_type):
            try:
                self.send_response(status)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                # The browser may cancel an image request while scrubbing.
                pass

        def do_GET(self):
            parsed = urlsplit(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path in static_files:
                    body, content_type = static_files[parsed.path]
                    self.reply(200, body, content_type)
                    return
                if parsed.path == '/api/dataset':
                    payload = {'episodes': list(episodes), 'anomalies': anomalies}
                elif parsed.path == '/api/episode':
                    episode_id = query['id'][0]
                    payload = read_episode(root, episodes[episode_id])
                    if any(record['end_ns'] > payload['end_ns'] for record in (anomalies or [])
                           if record['episode_id'] == episode_id):
                        raise ValueError(f'Anomaly interval exceeds episode duration: {episode_id}')
                elif parsed.path == '/api/frame':
                    episode_id = query['id'][0]
                    camera = query['camera'][0]
                    if camera not in episodes[episode_id]['cameras']:
                        raise KeyError(camera)
                    index = int(query['index'][0])
                    if index < 0:
                        raise ValueError('Negative frame index')
                    width = int(query.get('width', ['1280'])[0])
                    if not 64 <= width <= 2048:
                        raise ValueError('Invalid frame width')
                    path = root / 'episodes' / episode_id / 'rgb' / f'{camera}.mp4'
                    with readers_lock:
                        reader = readers.get(path)
                        if reader is None:
                            reader = readers[path] = FrameReader()
                    self.reply(200, reader.get(path, index, width), 'image/jpeg')
                    return
                else:
                    self.reply(404, b'Not found', 'text/plain; charset=utf-8')
                    return
                self.reply(200, json.dumps(payload, ensure_ascii=False).encode('utf-8'),
                           'application/json; charset=utf-8')
            except (KeyError, IndexError, ValueError, FileNotFoundError,
                    StopIteration, ImportError) as exc:
                self.reply(400, str(exc).encode('utf-8'), 'text/plain; charset=utf-8')

    try:
        with ThreadingHTTPServer((host, port), Handler) as server:
            print(f'Export viewer: http://{host}:{server.server_port}/')
            server.serve_forever()
    finally:
        for reader in readers.values():
            reader.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path, help='Exported dataset root')
    parser.add_argument('--anomalies', type=Path,
                        help='Anomaly JSONL report (default: <dataset>/anomalies.jsonl if present)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    try:
        serve(args.dataset.resolve(), args.host, args.port, args.anomalies)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
