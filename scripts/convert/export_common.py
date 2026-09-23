"""Shared writers for the export contract."""
from fractions import Fraction
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def relative_times(timestamps, origin):
    values = [int(t) - int(origin) for t in timestamps]
    if not values or values[0] < 0 or any(b < a for a, b in zip(values, values[1:])):
        raise ValueError('Empty, negative or unordered sample timestamps')
    return pa.array(values, type=pa.int64())


def write_state(path, timestamps, values, origin, *, pose=False):
    name = 'pose' if pose else 'openness'
    dtype = pa.list_(pa.float64(), 9) if pose else pa.float32()
    data = np.asarray(values, dtype=np.float64)
    if not np.isfinite(data).all():
        raise ValueError(f'Nonfinite {name}: {path}')
    if pose:
        if data.shape != (len(timestamps), 9):
            raise ValueError('Expected N x 9 poses')
        a, b = data[:, 3:6], data[:, 6:9]
        if not (np.allclose(np.sum(a*a, axis=1), 1, atol=1e-6)
                and np.allclose(np.sum(b*b, axis=1), 1, atol=1e-6)
                and np.allclose(np.sum(a*b, axis=1), 0, atol=1e-6)):
            raise ValueError('Invalid rotation6D')
    elif data.shape != (len(timestamps),) or np.any((data < 0) | (data > 1)):
        raise ValueError('Expected scalar gripper openness in [0,1]')
    schema = pa.schema([pa.field('timestamp_ns', pa.int64(), nullable=False),
                        pa.field(name, dtype, nullable=False)])
    table = pa.Table.from_arrays([relative_times(timestamps, origin),
                                 pa.array(data.tolist(), type=dtype)], schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def write_video_index(path, timestamps, origin):
    schema = pa.schema([pa.field('frame_index', pa.int64(), nullable=False),
                        pa.field('timestamp_ns', pa.int64(), nullable=False)])
    pq.write_table(pa.Table.from_arrays([
        pa.array(range(len(timestamps)), type=pa.int64()),
        relative_times(timestamps, origin)], schema=schema), path)


def create_video_decoder(codec, backend='cpu', *, keep_on_device=False):
    import av
    if codec not in ('h264', 'h265'):
        raise ValueError(f'Unsupported source video codec: {codec}')
    if backend not in ('cpu', 'nvdec'):
        raise ValueError(f'Unknown video decoder: {backend}')
    if keep_on_device and backend != 'nvdec':
        raise ValueError('Device frames require NVDEC')
    kwargs = {}
    if backend == 'nvdec':
        from av.codec.hwaccel import HWAccel
        kwargs['hwaccel'] = HWAccel('cuda', allow_software_fallback=False,
                                    is_hw_owned=keep_on_device)
    decoder = av.CodecContext.create('hevc' if codec == 'h265' else codec, 'r', **kwargs)
    decoder.open()
    if backend == 'nvdec' and not decoder.is_hwaccel:
        raise RuntimeError('NVDEC requested but hardware decoding is not active')
    return decoder


class VideoWriter:
    """Decode one Annex B access unit per sample, preserving PTS through reorder.

    MP4 uses an arbitrary 30 Hz container clock; actual times live in Parquet.
    Only compressed samples/timestamps are queued, never a whole decoded video.
    """
    def __init__(self, path: Path, codec: str, *, decoder_backend='cpu', encoder_backend='libx264'):
        import av
        if codec not in ('h264', 'h265'):
            raise ValueError(f'Unsupported source video codec: {codec}')
        if encoder_backend not in ('libx264', 'nvenc'):
            raise ValueError(f'Unknown video encoder: {encoder_backend}')
        if encoder_backend == 'nvenc' and decoder_backend != 'nvdec':
            raise ValueError('NVENC path requires NVDEC for GPU-resident frames')
        self.encoder_backend = encoder_backend
        self.device_frames = 0
        self.codec = codec
        self.decoder = create_video_decoder(codec, decoder_backend,
                                            keep_on_device=encoder_backend == 'nvenc')
        self.output = av.open(str(path), 'w')
        self.stream = None
        self.source_times = []
        self.timestamps = []
        self.seen = set()

    def add(self, data, timestamp, codec):
        import av
        if codec != self.codec or not data:
            raise ValueError('Video codec changed or empty access unit')
        packet = av.Packet(data)
        packet.pts = len(self.source_times)
        packet.time_base = Fraction(1, 30)
        self.source_times.append(timestamp)
        for frame in self.decoder.decode(packet):
            self._frame(frame)

    def _frame(self, frame):
        if self.encoder_backend == 'nvenc':
            if frame.format.name != 'cuda':
                raise RuntimeError('NVDEC → NVENC requires CUDA frames; refusing host-frame fallback')
            self.device_frames += 1
        index = frame.pts
        if index is None or index in self.seen or not 0 <= index < len(self.source_times):
            raise ValueError('Cannot map decoded frame to a unique MCAP sample')
        timestamp = self.source_times[index]
        if self.timestamps and timestamp < self.timestamps[-1]:
            raise ValueError('Video display order has decreasing acquisition timestamps')
        self.seen.add(index)
        if self.stream is None:
            hardware = self.encoder_backend == 'nvenc'
            self.stream = self.output.add_stream('h264_nvenc' if hardware else 'libx264', rate=30)
            self.stream.width, self.stream.height = frame.width, frame.height
            self.stream.pix_fmt = 'cuda' if hardware else 'yuv420p'
            if hardware:
                # PyAV adopts the first CUDA frame's hw_frames_ctx before opening
                # NVENC. No reformat/download/upload and no separate CUDA device.
                self.stream.bit_rate = 0
                self.stream.options = {'preset': 'p4', 'rc': 'vbr', 'cq': '18',
                                       'bf': '0', 'rc-lookahead': '0', 'delay': '0'}
            else:
                self.stream.options = {'crf': '18', 'preset': 'fast'}
        elif (frame.width, frame.height) != (self.stream.width, self.stream.height):
            raise ValueError('Video resolution changed within episode')
        frame.pts = len(self.timestamps)
        frame.time_base = Fraction(1, 30)
        self.timestamps.append(timestamp)
        for packet in self.stream.encode(frame):
            self.output.mux(packet)

    def finish(self):
        for frame in self.decoder.decode(None):
            self._frame(frame)
        if not self.timestamps or len(self.timestamps) != len(self.source_times):
            raise ValueError('Video frame count differs from MCAP sample count')
        for packet in self.stream.encode():
            self.output.mux(packet)
        self.output.close()

    def close(self):
        self.output.close()
