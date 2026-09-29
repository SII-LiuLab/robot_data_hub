"""Validate a completed staging episode before it becomes resumable."""
import re

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

STATE_NAMES = ('left_eef', 'right_eef', 'left_gripper', 'right_gripper')


def validate_record(record, episode_id):
    if not isinstance(record, dict) or record.get('episode_id') != episode_id:
        raise ValueError(f'{episode_id}: invalid manifest identity')
    cameras = record.get('cameras')
    if (not isinstance(cameras, list) or not cameras
            or any(not isinstance(c, str) or not re.fullmatch(r'[a-z0-9_]+', c) for c in cameras)
            or len(set(cameras)) != len(cameras)):
        raise ValueError(f'{episode_id}: invalid cameras')
    instructions = record.get('instructions')
    if not isinstance(instructions, list) or not instructions:
        raise ValueError(f'{episode_id}: missing instructions')
    previous = 0
    for instruction in instructions:
        if not isinstance(instruction, dict):
            raise ValueError(f'{episode_id}: invalid instruction')
        start, end, text = (instruction.get(k) for k in ('start_ns', 'end_ns', 'text'))
        if (type(start) is not int or type(end) is not int
                or not previous <= start < end < 2**63
                or not isinstance(text, str) or not text.strip()):
            raise ValueError(f'{episode_id}: invalid instruction interval/text')
        previous = end


def file_sizes(directory, record):
    """Cheap resume check; committed media are not decoded again."""
    expected = {f'state/{name}.parquet' for name in STATE_NAMES}
    expected.update(f'rgb/{camera}.{ext}' for camera in record['cameras'] for ext in ('mp4', 'parquet'))
    if (not directory.is_dir() or directory.is_symlink()
            or {p.name for p in directory.iterdir()} != {'state', 'rgb'}):
        raise ValueError(f'{directory}: invalid episode directory')
    files = {}
    for kind in ('state', 'rgb'):
        folder = directory / kind
        if not folder.is_dir() or folder.is_symlink():
            raise ValueError(f'{folder}: invalid stream directory')
        for path in folder.iterdir():
            if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
                raise ValueError(f'{path}: invalid stream file')
            files[path.relative_to(directory).as_posix()] = path.stat().st_size
    if set(files) != expected:
        raise ValueError(f'{directory}: stream files differ from manifest')
    return files


def validate_episode(directory, record):
    validate_record(record, directory.name)
    sizes = file_sizes(directory, record)
    first, last = [], []
    for relative in sorted(sizes):
        if not relative.endswith('.parquet'):
            continue
        path = directory / relative
        table = pq.read_table(path)
        video = relative.startswith('rgb/')
        pose = relative.endswith('_eef.parquet')
        field = 'frame_index' if video else ('pose' if pose else 'openness')
        dtype = pa.int64() if video else (pa.list_(pa.float64(), 9) if pose else pa.float32())
        if (set(table.column_names) != {'timestamp_ns', field}
                or table.schema.field('timestamp_ns').type != pa.int64()
                or table.schema.field(field).type != dtype
                or any(c.null_count for c in table.columns) or not table.num_rows
                or (pose and table.schema.field('pose').nullable)):
            raise ValueError(f'{path}: invalid schema or empty/null samples')
        times = table['timestamp_ns'].to_numpy()
        if times[0] < 0 or np.any(times[1:] < times[:-1]):
            raise ValueError(f'{path}: invalid timestamps')
        first.append(int(times[0]))
        last.append(int(times[-1]))
        values = np.asarray(table[field].to_pylist())
        if not np.isfinite(values).all():
            raise ValueError(f'{path}: nonfinite values')
        if video:
            if not np.array_equal(values, np.arange(len(times))):
                raise ValueError(f'{path}: invalid frame indices')
            # Converters already decode every input frame and flush the encoder.
            # Verify the resulting MP4 sample count without decoding it a second time.
            with av.open(str(path.with_suffix('.mp4'))) as reader:
                if (len(reader.streams) != 1 or len(reader.streams.video) != 1
                        or reader.streams.video[0].codec_context.name != 'h264'
                        or 'mp4' not in reader.format.name.split(',')
                        or reader.streams.video[0].frames != len(times)):
                    raise ValueError(f'{path}: video format/frame count differs from index')
        elif pose:
            a, b = values[:, 3:6], values[:, 6:9]
            if not (np.allclose(np.sum(a*a, axis=1), 1, atol=1e-6)
                    and np.allclose(np.sum(b*b, axis=1), 1, atol=1e-6)
                    and np.allclose(np.sum(a*b, axis=1), 0, atol=1e-6)):
                raise ValueError(f'{path}: invalid rotation6D')
        elif np.any((values < 0) | (values > 1)):
            raise ValueError(f'{path}: invalid gripper openness')
    if min(first) != 0 or record['instructions'][-1]['end_ns'] > max(last):
        raise ValueError(f'{directory}: invalid shared origin/instruction extent')
    return sizes
