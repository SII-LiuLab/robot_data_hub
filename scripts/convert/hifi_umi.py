#!/usr/bin/env python3
"""Convert HiFi-UMI-2K local LeRobot v3 shards or an extracted episode subset."""
from __future__ import annotations

import argparse
import json
from fractions import Fraction
import math
from pathlib import Path
import re
import tempfile

import numpy as np
import pyarrow.parquet as pq

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.convert.export_common import (
    configure_codec, copy_h264_video, seconds_to_ns, write_state, write_video_index,
    task_instructions as instructions,
)

ROOT = Path(__file__).resolve().parents[2]
# Dataset convention: the stored angle is the single-finger angle, not doubled.
DEFAULT_GRIPPER_OPEN_RAD = float(np.deg2rad(35.))
CAMERAS = {f'observation.images.{name}': name for name in (
    'head_main', 'head_main_stereo_right', 'left_hand_up', 'left_hand_down',
    'right_hand_up', 'right_hand_down',
)}
COLUMNS = ('observation.state', 'observation.state_valid', 'timestamp',
           'frame_index', 'episode_index', 'index', 'task_index', 'valid.frame')


def validate_source(info, modality):
    """Refuse other embodiments/layouts instead of guessing their semantics."""
    if info.get('robot_type') != 'simpleai_umi_4.0' or info.get('codebase_version') != 'v3.0':
        raise ValueError('Expected HiFi-UMI simpleai_umi_4.0 LeRobot v3.0')
    layout = info.get('state_layout', {})
    expected = {'feature': 'observation.state', 'order': ['right', 'left'],
                'rotation_6d_layout': 'first_two_rows', 'gripper_unit': 'rad',
                'per_side': ['x', 'y', 'z', *(f'rot6d_{i}' for i in range(6)),
                             'gripper_angle_rad']}
    if any(layout.get(key) != value for key, value in expected.items()):
        raise ValueError('Unsupported HiFi-UMI state layout')
    features = info.get('features', {})
    if features.get('observation.state', {}).get('shape') != [20]:
        raise ValueError('Expected a 20-dimensional observation.state')
    if not all(features.get(key, {}).get('dtype') == 'video' for key in CAMERAS):
        raise ValueError('Expected all six HiFi-UMI video features')
    for side, offset in (('right', 0), ('left', 10)):
        for kind, start, end in (('pose', offset, offset + 9),
                                 ('gripper', offset + 9, offset + 10)):
            field = modality.get('state', {}).get(f'{side}_eef_{kind}', {})
            expected = {'start': start, 'end': end, 'absolute': True,
                        'original_key': 'observation.state'}
            expected.update({'rotation_type': 'rotation_6d',
                             'rotation_6d_layout': 'first_two_rows'} if kind == 'pose'
                            else {'unit': 'rad'})
            if any(field.get(key) != value for key, value in expected.items()):
                raise ValueError(f'Unsupported modality for {side} {kind}')


def poses_from_rows(values):
    """Source TCP/axes already match the contract; only change the 6D layout."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 9 or not np.isfinite(values).all():
        raise ValueError('Expected finite N x 9 source poses')
    first, second = values[:, 3:6].copy(), values[:, 6:9].copy()
    if not (np.allclose(np.sum(first * first, axis=1), 1, atol=1e-5, rtol=0)
            and np.allclose(np.sum(second * second, axis=1), 1, atol=1e-5, rtol=0)
            and np.allclose(np.sum(first * second, axis=1), 0, atol=1e-5, rtol=0)):
        raise ValueError('Invalid source rotation6D rows')
    # Remove float32 roundoff only, after rejecting malformed rotations.
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    second -= np.sum(first * second, axis=1, keepdims=True) * first
    second /= np.linalg.norm(second, axis=1, keepdims=True)
    rotation = np.stack((first, second, np.cross(first, second)), axis=1)
    return np.concatenate((values[:, :3], rotation[:, :, 0], rotation[:, :, 1]), axis=1)


def validate_gripper_range(closed_rad, open_rad):
    if not np.isfinite([closed_rad, open_rad]).all() or open_rad <= closed_rad:
        raise ValueError('Gripper range requires finite open_rad > closed_rad')


def openness(angles, closed_rad=0., open_rad=DEFAULT_GRIPPER_OPEN_RAD):
    validate_gripper_range(closed_rad, open_rad)
    angles = np.asarray(angles, dtype=np.float64)
    if not np.isfinite(angles).all():
        raise ValueError('Nonfinite gripper angle')
    return np.clip((angles - closed_rad) / (open_rad - closed_rad), 0, 1)


def load_tasks(path):
    tasks = {}
    for row in pq.read_table(path, columns=['task_index', 'task']).to_pylist():
        index, text = row['task_index'], row['task']
        if (type(index) is not int or index in tasks
                or not isinstance(text, str) or not text.strip()):
            raise ValueError('Invalid/duplicate task_index or empty task text')
        tasks[index] = text
    return tasks


def read_episode(source, tasks, closed_rad, open_rad):
    metadata = json.loads((source / 'episode.json').read_text(encoding='utf-8'))
    rows = pq.read_table(source / 'data.parquet', columns=list(COLUMNS)).to_pydict()
    index = metadata.get('episode_index')
    if type(index) is not int or source.name != f'episode_{index:06d}':
        raise ValueError(f'{source}: inconsistent episode identity')
    return read_rows(source, metadata, rows, tasks, closed_rad, open_rad)


def read_rows(source, metadata, rows, tasks, closed_rad, open_rad):
    count = len(rows['timestamp'])
    index = metadata.get('episode_index')
    start, stop = metadata.get('dataset_from_index'), metadata.get('dataset_to_index')
    if (type(index) is not int or index < 0
            or type(start) is not int or type(stop) is not int or start < 0
            or count == 0 or metadata.get('length') != count or stop - start != count
            or any(i != index for i in rows['episode_index'])):
        raise ValueError(f'{source}: inconsistent episode identity/length')
    if (rows['frame_index'] != list(range(count))
            or rows['index'] != list(range(start, stop))):
        raise ValueError(f'{source}: noncontiguous frame/index; cannot map video frames')
    if any(t is None for t in rows['timestamp']):
        raise ValueError(f'{source}: null timestamp')
    times = [seconds_to_ns(t) for t in rows['timestamp']]
    if times[0] < 0 or any(b < a for a, b in zip(times, times[1:])):
        raise ValueError(f'{source}: negative or decreasing timestamps')
    if any(value is not True for value in rows['valid.frame']):
        raise ValueError(f'{source}: valid.frame contains invalid samples')
    valid = rows['observation.state_valid']
    if any(row is None or len(row) != 20 or any(v is not True for v in row) for row in valid):
        raise ValueError(f'{source}: observation.state_valid contains invalid samples')
    state = np.asarray(rows['observation.state'], dtype=np.float64)
    if state.shape != (count, 20) or not np.isfinite(state).all():
        raise ValueError(f'{source}: expected finite N x 20 observation.state')
    annotation = instructions(rows['task_index'], tasks, times)
    if set(metadata.get('tasks', [])) != {tasks[i] for i in rows['task_index']}:
        raise ValueError(f'{source}: episode tasks disagree with task_index')
    if set(metadata.get('videos', {})) != set(CAMERAS):
        raise ValueError(f'{source}: expected all six camera views')
    streams = {}
    for side, offset in (('right', 0), ('left', 10)):
        streams[f'{side}_eef'] = poses_from_rows(state[:, offset:offset + 9])
        streams[f'{side}_gripper'] = openness(state[:, offset + 9], closed_rad, open_rad)
    return times, streams, annotation


def convert_episode(source, destination, tasks, closed_rad, open_rad):
    times, streams, annotation = read_episode(source, tasks, closed_rad, open_rad)
    return write_episode(source.name, destination, times, streams, annotation,
                         lambda key, target: copy_h264_video(
                             source / 'videos' / f'{key}.mp4', target, len(times)))


def write_episode(episode_id, destination, times, streams, annotation, video):
    for name, values in streams.items():
        write_state(destination / 'state' / f'{name}.parquet', times, values, times[0],
                    pose=name.endswith('_eef'))
    for key, camera in CAMERAS.items():
        print(f'  {camera}', flush=True)
        target = destination / 'rgb' / camera
        video(key, target.with_suffix('.mp4'))
        write_video_index(target.with_suffix('.parquet'), times, times[0])
    return {'episode_id': episode_id, 'cameras': sorted(CAMERAS.values()),
            'instructions': annotation}


def discover_shards(source):
    if (source / 'meta/info.json').is_file():
        shards = [source]
    else:
        shards = sorted(p for p in source.glob('chunk-*/part-*') if p.is_dir())
    if not shards:
        raise ValueError(f'{source}: expected chunk-XXXX/part-YYYY shards or source/ and episodes/ subset')
    for shard in shards:
        if not (re.fullmatch(r'chunk-\d+', shard.parent.name)
                and re.fullmatch(r'part-\d+', shard.name)):
            raise ValueError(f'Invalid HiFi-UMI shard name: {shard}')
    return shards


def shard_episode_id(shard, index):
    return f'{shard.parent.name}_{shard.name}_episode_{index:06d}'.replace('-', '_')


def shard_path(shard, template, row, prefix, video_key=None):
    indices = [row[f'{prefix}/{name}_index'] for name in ('chunk', 'file')]
    if any(type(i) is not int or i < 0 for i in indices):
        raise ValueError(f'{shard}: invalid {prefix} chunk/file indices')
    path = shard / template.format(chunk_index=indices[0], file_index=indices[1], video_key=video_key)
    if not path.resolve().is_relative_to(shard.resolve()):
        raise ValueError(f'{shard}: source path escapes shard: {path}')
    return path


class ShardReader:
    """Keep metadata and at most one projected frame table for the current shard."""

    def __init__(self, source):
        self.source = source
        meta = source / 'meta'
        self.info = json.loads((meta / 'info.json').read_text(encoding='utf-8'))
        validate_source(self.info, json.loads((meta / 'modality.json').read_text(encoding='utf-8')))
        self.tasks = load_tasks(meta / 'tasks.parquet')
        self.fps = self.info.get('fps')
        if not isinstance(self.fps, (float, int)) or not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError(f'{source}: invalid FPS')
        self.metadata_files = sorted((meta / 'episodes').rglob('*.parquet'))
        if not self.metadata_files:
            raise ValueError(f'{source}: no episode metadata parquet files')
        columns = ['episode_index', 'length', 'dataset_from_index', 'dataset_to_index', 'tasks',
                   'data/chunk_index', 'data/file_index']
        columns += [f'videos/{key}/{field}' for key in CAMERAS for field in
                    ('chunk_index', 'file_index', 'from_timestamp', 'to_timestamp')]
        self.episodes = {}
        self.paths = {}
        for path in self.metadata_files:
            for row in pq.read_table(path, columns=columns).to_pylist():
                index, length = row['episode_index'], row['length']
                start, stop = row['dataset_from_index'], row['dataset_to_index']
                if (type(index) is not int or index < 0 or index in self.episodes
                        or type(length) is not int or length < 1
                        or type(start) is not int or type(stop) is not int
                        or start < 0 or stop - start != length):
                    raise ValueError(f'{path}: invalid/duplicate episode identity or length')
                videos = {}
                for key in CAMERAS:
                    prefix = f'videos/{key}'
                    begin, end = row[f'{prefix}/from_timestamp'], row[f'{prefix}/to_timestamp']
                    if (not isinstance(begin, (int, float)) or not isinstance(end, (int, float))
                            or not math.isfinite(begin) or not math.isfinite(end) or begin < 0
                            or end <= begin or abs((end - begin) * self.fps - length) > .01):
                        raise ValueError(f'{path}: invalid video range for episode {index}: {key}')
                    videos[key] = {'path': self.path(self.info['video_path'], row, prefix, key),
                                   'from_timestamp': begin, 'to_timestamp': end}
                row['videos'] = videos
                row['data_path'] = self.path(self.info['data_path'], row, 'data')
                self.episodes[index] = row
        if not self.episodes or len(self.episodes) != self.info.get('total_episodes'):
            raise ValueError(f'{source}: episode metadata count disagrees with info.json')
        self.data_path = self.table = None

    def path(self, template, row, prefix, video_key=None):
        indices = (row[f'{prefix}/chunk_index'], row[f'{prefix}/file_index'])
        if any(type(i) is not int or i < 0 for i in indices):
            raise ValueError(f'{self.source}: invalid {prefix} chunk/file indices')
        key = (template, *indices, video_key)
        if key not in self.paths:
            self.paths[key] = str(shard_path(self.source, template, row, prefix, video_key))
        return self.paths[key]

    def inputs(self, indices):
        paths = {str(p) for p in self.metadata_files}
        for index in indices:
            row = self.episodes[index]
            paths.add(row['data_path'])
            paths.update(v['path'] for v in row['videos'].values())
        return [Path(p) for p in sorted(paths)]

    def convert_episode(self, index, destination, closed_rad, open_rad):
        row = self.episodes[index]
        path = row['data_path']
        if path != self.data_path:
            # Release the preceding file before loading another, even across file boundaries.
            self.table = None
            self.data_path = None
            self.table = pq.read_table(path, columns=list(COLUMNS))
            self.data_path = path
        if len(self.table) == 0:
            raise ValueError(f'{path}: empty frame table')
        start = row['dataset_from_index'] - self.table['index'][0].as_py()
        if start < 0 or start + row['length'] > len(self.table):
            raise ValueError(f'{path}: episode {index} range is outside its data file')
        rows = self.table.slice(start, row['length']).to_pydict()
        times, streams, annotation = read_rows(self.source, row, rows, self.tasks, closed_rad, open_rad)
        episode_id = shard_episode_id(self.source, index)
        return write_episode(episode_id, destination, times, streams, annotation,
                             lambda key, target: extract_video_segment(
                                 row['videos'][key], target, len(times), self.fps))


def extract_video_segment(video, destination, expected_frames, fps):
    """Decode from a preceding keyframe, select display frames, and encode H.264."""
    import av

    start, end = video['from_timestamp'], video['to_timestamp']
    rate = Fraction(str(fps))
    tolerance = min(1e-3, .01 / fps)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with av.open(video['path']) as reader, av.open(str(destination), 'w') as writer:
        if len(reader.streams.video) != 1:
            raise ValueError(f'{video["path"]}: expected one video stream')
        source = reader.streams.video[0]
        configure_codec(source.codec_context)
        reader.seek(math.floor(start / source.time_base), stream=source, backward=True, any_frame=False)
        encoder = writer.add_stream('libx264', rate=rate)
        encoder.width, encoder.height = source.width, source.height
        encoder.pix_fmt = 'yuv420p'
        encoder.options = {'crf': '18', 'preset': 'fast'}
        configure_codec(encoder.codec_context)
        count = 0
        for frame in reader.decode(source):
            timestamp = frame.time
            if timestamp is None:
                raise ValueError(f'{video["path"]}: video frame has no timestamp')
            if timestamp < start - tolerance:
                continue
            if timestamp >= end - tolerance:
                break
            if (frame.width, frame.height) != (encoder.width, encoder.height):
                raise ValueError(f'{video["path"]}: video resolution changed')
            # Reject missing/extra frames even when their total count happens to match.
            if abs(timestamp - (start + count / fps)) > tolerance:
                raise ValueError(f'{video["path"]}: noncontiguous video segment at frame {count}')
            frame = frame.reformat(format='yuv420p')
            frame.pts, frame.time_base = count, 1 / rate
            for packet in encoder.encode(frame):
                writer.mux(packet)
            count += 1
        if count != expected_frames:
            raise ValueError(f'{video["path"]}: expected {expected_frames} frames, decoded {count}')
        for packet in encoder.encode():
            writer.mux(packet)


def convert(source, output, *, gripper_open_rad=DEFAULT_GRIPPER_OPEN_RAD,
            gripper_closed_rad=0., limit=None):
    validate_gripper_range(gripper_closed_rad, gripper_open_rad)
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('limit must be a positive integer')
    if output.exists():
        raise ValueError(f'Output already exists; choose a new directory: {output}')
    if not (source / 'source').is_dir():
        return convert_shards(source, output, gripper_closed_rad, gripper_open_rad, limit)
    validate_source(json.loads((source / 'source/info.json').read_text(encoding='utf-8')),
                    json.loads((source / 'source/modality.json').read_text(encoding='utf-8')))
    tasks = load_tasks(source / 'source/tasks.parquet')
    episodes = sorted(p for p in (source / 'episodes').iterdir() if p.is_dir())
    if not episodes or any(not re.fullmatch(r'episode_\d+', p.name) for p in episodes):
        raise ValueError(f'No episodes or invalid episode directory names under {source}')
    episodes = episodes[:limit]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as temporary:
        staged = Path(temporary) / 'dataset'
        staged.mkdir()
        with (staged / 'episodes.jsonl').open('w', encoding='utf-8') as handle:
            for i, episode in enumerate(episodes, 1):
                print(f'[{i}/{len(episodes)}] {episode.name}', flush=True)
                record = convert_episode(episode, staged / 'episodes' / episode.name,
                                         tasks, gripper_closed_rad, gripper_open_rad)
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        if output.exists():
            raise ValueError(f'Output appeared during conversion: {output}')
        staged.rename(output)
    print(f'Exported {len(episodes)} episodes to {output}', flush=True)


def convert_shards(source, output, closed_rad, open_rad, limit):
    shards = discover_shards(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with tempfile.TemporaryDirectory(prefix=f'.{output.name}-', dir=output.parent) as temporary:
        staged = Path(temporary) / 'dataset'
        staged.mkdir()
        with (staged / 'episodes.jsonl').open('w', encoding='utf-8') as handle:
            for shard in shards:
                reader = ShardReader(shard)
                remaining = None if limit is None else limit - count
                for index in sorted(reader.episodes)[:remaining]:
                    episode_id = shard_episode_id(shard, index)
                    print(f'[{count + 1}] {episode_id}', flush=True)
                    record = reader.convert_episode(index, staged / 'episodes' / episode_id, closed_rad, open_rad)
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                    count += 1
                # Release the frame table before indexing/loading the next shard.
                del reader
                if limit is not None and count >= limit:
                    break
        if output.exists():
            raise ValueError(f'Output appeared during conversion: {output}')
        staged.rename(output)
    print(f'Exported {count} episodes to {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'dataset/raw/HiFi-UMI-2K')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'dataset/processed/HiFi-UMI-2K')
    parser.add_argument('--limit', type=int, help='convert only the first N sorted episodes')
    parser.add_argument('--gripper-closed-rad', type=float, default=0.,
                        help='fixed dataset-wide closed angle (default: 0)')
    parser.add_argument('--gripper-open-rad', type=float, default=DEFAULT_GRIPPER_OPEN_RAD,
                        help='fully open single-finger angle (default: 35 degrees = 0.6108652382 rad)')
    args = parser.parse_args()
    import av
    try:
        convert(args.input_dir, args.output_dir, gripper_open_rad=args.gripper_open_rad,
                gripper_closed_rad=args.gripper_closed_rad, limit=args.limit)
    except (ValueError, OSError, av.FFmpegError) as exc:
        parser.exit(1, f'Error: {exc}\n')


if __name__ == '__main__':
    main()
