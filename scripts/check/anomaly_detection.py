"""Detect anomalies in an exported dataset without resampling state streams."""

import argparse
from dataclasses import dataclass, fields
import json
import math
from pathlib import Path
import re
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


STATE_NAMES = ('left_eef', 'right_eef', 'left_gripper', 'right_gripper')
NS_PER_SECOND = 1_000_000_000


@dataclass(frozen=True)
class Config:
    position_tolerance_m: float = .01
    orientation_tolerance_deg: float = 5.
    openness_tolerance: float = .05
    max_boundary_idle_s: float = 3.
    position_acceleration_limit: float = 100.
    orientation_acceleration_limit: float = 20000.
    openness_acceleration_limit: float = 500.
    max_internal_idle_s: float = 10.

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{field.name} must be finite and nonnegative')


def rotation_quaternions(poses):
    """Contract rotation6D (first two columns) to unit wxyz quaternions."""
    a, b = poses[:, 3:6].copy(), poses[:, 6:9].copy()
    if not (np.allclose(np.sum(a*a, axis=1), 1, atol=1e-6)
            and np.allclose(np.sum(b*b, axis=1), 1, atol=1e-6)
            and np.allclose(np.sum(a*b, axis=1), 0, atol=1e-6)):
        raise ValueError('Invalid rotation6D')
    # Remove floating point drift before calculating angles and SLERP.
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b -= np.sum(a*b, axis=1, keepdims=True) * a
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    r = np.stack((a, b, np.cross(a, b)), axis=-1)
    r00, r11, r22 = r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]
    squared = np.column_stack((1+r00+r11+r22, 1+r00-r11-r22,
                               1-r00+r11-r22, 1-r00-r11+r22))
    candidates = np.stack((
        np.column_stack((squared[:, 0], r[:, 2, 1]-r[:, 1, 2],
                         r[:, 0, 2]-r[:, 2, 0], r[:, 1, 0]-r[:, 0, 1])),
        np.column_stack((r[:, 2, 1]-r[:, 1, 2], squared[:, 1],
                         r[:, 1, 0]+r[:, 0, 1], r[:, 0, 2]+r[:, 2, 0])),
        np.column_stack((r[:, 0, 2]-r[:, 2, 0], r[:, 1, 0]+r[:, 0, 1],
                         squared[:, 2], r[:, 2, 1]+r[:, 1, 2])),
        np.column_stack((r[:, 1, 0]-r[:, 0, 1], r[:, 0, 2]+r[:, 2, 0],
                         r[:, 2, 1]+r[:, 1, 2], squared[:, 3])),
    ), axis=1)
    q = candidates[np.arange(len(poses)), np.argmax(squared, axis=1)]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def rotation_distance_deg(a, b):
    b = np.where(np.sum(a*b, axis=-1, keepdims=True) < 0, -b, b)
    return np.degrees(4 * np.arctan2(np.linalg.norm(a-b, axis=-1),
                                    np.linalg.norm(a+b, axis=-1)))


def slerp(a, b, fraction):
    dot = np.sum(a*b, axis=1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    angle = np.arccos(np.clip(np.abs(dot), 0, 1)) / np.pi
    fraction = fraction[:, None]
    q = ((1-fraction)*np.sinc((1-fraction)*angle)*a
         + fraction*np.sinc(fraction*angle)*b) / np.sinc(angle)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


@dataclass
class StateStream:
    times: np.ndarray
    values: np.ndarray
    rotations: np.ndarray | None = None

    def changed(self, left, right, config):
        if self.rotations is None:
            return np.abs(self.values[left] - self.values[right]) > config.openness_tolerance
        return ((np.linalg.norm(self.values[left, :3] - self.values[right, :3], axis=-1)
                 > config.position_tolerance_m)
                | (rotation_distance_deg(self.rotations[left], self.rotations[right])
                   > config.orientation_tolerance_deg))


def read_times(table, path):
    column = table['timestamp_ns']
    if column.type != pa.int64() or column.null_count:
        raise ValueError(f'{path}: timestamps must be non-null int64')
    times = column.to_numpy()
    if not len(times) or times[0] < 0 or np.any(times[1:] < times[:-1]):
        raise ValueError(f'{path}: empty, negative or unordered timestamps')
    return times


def read_episode(root, record):
    episode_id = record['episode_id']
    if (not isinstance(episode_id, str) or episode_id in ('', '.', '..')
            or '/' in episode_id or '\\' in episode_id):
        raise ValueError(f'Invalid episode_id: {episode_id!r}')
    base = root / 'episodes' / episode_id
    streams = {}
    for name in STATE_NAMES:
        pose = name.endswith('_eef')
        field = 'pose' if pose else 'openness'
        path = base / 'state' / f'{name}.parquet'
        table = pq.read_table(path, columns=['timestamp_ns', field])
        times = read_times(table, path)
        values = np.asarray(table[field].to_pylist(), dtype=np.float64)
        if (table[field].null_count or not np.isfinite(values).all()
                or values.shape != ((len(times), 9) if pose else (len(times),))):
            raise ValueError(f'{path}: invalid {field}')
        if not pose and np.any((values < 0) | (values > 1)):
            raise ValueError(f'{path}: openness outside [0,1]')
        streams[name] = StateStream(times, values, rotation_quaternions(values) if pose else None)
    cameras = record['cameras']
    if (not isinstance(cameras, list) or not cameras
            or any(not isinstance(c, str) or not re.fullmatch(r'[a-z0-9_]+', c) for c in cameras)
            or len(set(cameras)) != len(cameras)):
        raise ValueError(f'{episode_id}: invalid cameras')
    end_ns = max(int(stream.times[-1]) for stream in streams.values())
    for camera in cameras:
        path = base / 'rgb' / f'{camera}.parquet'
        times = read_times(pq.read_table(path, columns=['timestamp_ns']), path)
        end_ns = max(end_ns, int(times[-1]))
    return streams, end_ns


def detect_episode(index, streams, end_ns, config=Config()):
    """Return contract records in rule order; timestamps remain native int64 ns."""
    records = []

    def add(name, start, end, reason):
        records.append(dict(index=index, stream=name, start_ns=int(start),
                            end_ns=int(end), reason=reason))

    starts, tails = [], []
    changes = []
    for name in STATE_NAMES:
        stream = streams[name]
        first = np.flatnonzero(stream.changed(slice(None), 0, config))
        last = np.flatnonzero(stream.changed(slice(None), -1, config))
        starts.append(int(stream.times[first[0]]) if len(first) else end_ns)
        tails.append(end_ns - int(stream.times[last[-1]]) if len(last) else end_ns)
        changes.append(stream.times[1:][stream.changed(slice(1, None), slice(None, -1), config)])
    first, last = int(np.argmin(starts)), int(np.argmin(tails))
    if starts[first] / NS_PER_SECOND > config.max_boundary_idle_s:
        add(STATE_NAMES[first], 0, starts[first], 'boundary_idle')
    if tails[last] / NS_PER_SECOND > config.max_boundary_idle_s:
        add(STATE_NAMES[last], end_ns - tails[last], end_ns, 'boundary_idle')

    for name in STATE_NAMES:
        stream = streams[name]
        dt = np.diff(stream.times) / NS_PER_SECOND
        valid = (dt[:-1] > 0) & (dt[1:] > 0)
        middle = np.flatnonzero(valid) + 1
        prev, next_ = dt[:-1][valid], dt[1:][valid]
        fraction = prev / (prev + next_)
        factor = 2 / (prev * next_)
        values = stream.values
        if stream.rotations is None:
            predicted = (1-fraction)*values[middle-1] + fraction*values[middle+1]
            flagged = np.abs(values[middle]-predicted)*factor > config.openness_acceleration_limit
        else:
            predicted = ((1-fraction[:, None])*values[middle-1, :3]
                         + fraction[:, None]*values[middle+1, :3])
            acceleration = np.linalg.norm(values[middle, :3]-predicted, axis=1)*factor
            predicted_rotation = slerp(stream.rotations[middle-1], stream.rotations[middle+1], fraction)
            angular_acceleration = rotation_distance_deg(stream.rotations[middle], predicted_rotation)*factor
            flagged = ((acceleration > config.position_acceleration_limit)
                       | (angular_acceleration > config.orientation_acceleration_limit))
        for timestamp in stream.times[middle[flagged]]:
            add(name, timestamp, timestamp, 'state_jump')

    changes = np.unique(np.concatenate(changes))
    for left, right in zip(changes[:-1], changes[1:]):
        if (int(right) - int(left)) / NS_PER_SECOND > config.max_internal_idle_s:
            for name in STATE_NAMES:
                add(name, left, right, 'internal_idle')
    return records


def detect_dataset(root, output=None, config=Config()):
    """Process one episode at a time and atomically replace the JSONL report."""
    root = Path(root).resolve()
    output = Path(output).resolve() if output is not None else root / 'anomalies.jsonl'
    manifest = root / 'episodes.jsonl'
    if (output == manifest or output.suffix != '.jsonl'
            or output.is_relative_to(root / 'episodes')):
        raise ValueError('Output must be a .jsonl report outside episodes/ and cannot replace episodes.jsonl')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    episodes, anomalies = 0, 0
    seen = set()
    try:
        with manifest.open(encoding='utf-8') as source, tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8', dir=output.parent,
                prefix=f'.{output.name}.', suffix='.tmp', delete=False) as report:
            temporary = Path(report.name)
            for index, line in enumerate(source):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    streams, end_ns = read_episode(root, record)
                    if record['episode_id'] in seen:
                        raise ValueError(f'Duplicate episode_id: {record["episode_id"]}')
                    seen.add(record['episode_id'])
                    results = detect_episode(index, streams, end_ns, config)
                except (ValueError, KeyError, TypeError, OSError) as exc:
                    raise ValueError(f'{manifest}: line {index + 1}: {exc}') from exc
                for result in results:
                    report.write(json.dumps(result, separators=(',', ':')) + '\n')
                episodes += 1
                anomalies += len(results)
            if not episodes:
                raise ValueError('episodes.jsonl is empty')
        temporary.replace(output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return episodes, anomalies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path, help='Exported dataset root')
    parser.add_argument('--output', type=Path, help='JSONL report (default: DATASET/anomalies.jsonl)')
    for field in fields(Config):
        parser.add_argument('--' + field.name.replace('_', '-'), default=field.default,
                            type=field.type, help=f'Default: {field.default}')
    args = parser.parse_args()
    try:
        config = Config(**{field.name: getattr(args, field.name) for field in fields(Config)})
        episodes, anomalies = detect_dataset(args.dataset, args.output, config)
    except (OSError, ValueError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    print(f'Checked {episodes} episodes; wrote {anomalies} anomalies to '
          f'{args.output or args.dataset / "anomalies.jsonl"}')


if __name__ == '__main__':
    main()
