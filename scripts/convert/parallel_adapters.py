"""Source discovery and conversion adapters for the fixed parallel plan."""
from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
import re
import tempfile

ROOT = Path(__file__).resolve().parents[2]


class _NullProgress:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, step=1):
        pass


def _no_progress(label, total=None, *, unit='item'):
    return _NullProgress()


def module(dataset):
    return importlib.import_module(f'scripts.convert.{dataset}')


def make_config(dataset, source, *, model_dir=None, limit=None, limit_episodes=None,
                robot_type=None, tcp_offset=None, r1lite_tcp_offset=None, r1pro_tcp_offset=None,
                gripper_open_rad=None, gripper_closed_rad=None):
    source = Path(source).resolve()
    if not source.exists() or (dataset != 'agibotworld2026' and not source.is_dir()):
        raise ValueError(f'Input does not exist or has the wrong type: {source}')
    for value in (limit, limit_episodes):
        if value is not None and (type(value) is not int or value < 1):
            raise ValueError('Limits must be positive integers')
    if dataset not in ('abc130k', 'molmoact2', 'hifi_umi', 'galaxea', 'agibotworld2026'):
        raise ValueError(f'Unknown dataset: {dataset}')
    if dataset != 'galaxea' and any(x is not None for x in (robot_type, r1lite_tcp_offset, r1pro_tcp_offset)):
        raise ValueError('Robot type and Galaxea TCP overrides require --dataset galaxea')
    if tcp_offset is not None and dataset != 'agibotworld2026':
        raise ValueError('--tcp-offset requires --dataset agibotworld2026')
    if dataset != 'hifi_umi' and any(x is not None for x in (gripper_open_rad, gripper_closed_rad)):
        raise ValueError('Gripper angle overrides require --dataset hifi_umi')
    source_module = module(dataset)
    params = {}
    if dataset == 'hifi_umi':
        if model_dir is not None:
            raise ValueError('HiFi-UMI does not use a robot model')
        opened = source_module.DEFAULT_GRIPPER_OPEN_RAD if gripper_open_rad is None else gripper_open_rad
        closed = 0. if gripper_closed_rad is None else gripper_closed_rad
        source_module.validate_gripper_range(closed, opened)
        params.update(gripper_open_rad=opened, gripper_closed_rad=closed)
    else:
        default = {'abc130k': 'yam', 'molmoact2': 'yam', 'agibotworld2026': 'agibot_g2', 'galaxea': ''}[dataset]
        model_dir = str(Path(model_dir or ROOT / 'assets/robot_models' / default).resolve())
    if dataset == 'agibotworld2026':
        params['tcp_offset'] = list(tcp_offset if tcp_offset is not None else source_module.DEFAULT_TCP_OFFSET)
    if dataset == 'galaxea':
        if robot_type is not None and robot_type not in source_module.DEFAULT_TCP_OFFSETS:
            raise ValueError(f'Unsupported robot_type: {robot_type}')
        params['robot_type'] = robot_type
        params['tcp_offsets'] = {kind: list(offset if offset is not None else source_module.DEFAULT_TCP_OFFSETS[kind])
                                 for kind, offset in [('r1lite', r1lite_tcp_offset), ('r1pro', r1pro_tcp_offset)]}
    offsets = ([params['tcp_offset']] if 'tcp_offset' in params else []) + list(params.get('tcp_offsets', {}).values())
    if any(len(v) != 3 or not all(math.isfinite(x) for x in v) for v in offsets):
        raise ValueError('TCP offsets require three finite metres')
    return {'dataset': dataset, 'source': str(source), 'model_dir': model_dir,
            'limit': limit, 'limit_episodes': limit_episodes, 'params': params}


def configuration(args):
    names = ('model_dir', 'limit', 'limit_episodes', 'robot_type', 'tcp_offset',
             'r1lite_tcp_offset', 'r1pro_tcp_offset', 'gripper_open_rad', 'gripper_closed_rad')
    return make_config(args.dataset, args.input_dir, **{k: getattr(args, k) for k in names})


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def discover(config, progress=_no_progress):
    dataset, source = config['dataset'], Path(config['source'])
    m = module(dataset)
    tasks, shared = [], []
    task_limit = min((v for v in (config['limit'], config['limit_episodes']) if v is not None), default=None)
    if dataset == 'abc130k':
        found = []
        with progress('Scanning for episode.mcap files', unit='episode') as bar:
            for path in source.rglob('episode.mcap'):
                found.append(path)
                bar.update()
        for path in sorted(found)[:task_limit]:
            tasks.append({'source': str(path.parent), 'episode_ids': [path.parent.name],
                          'inputs': [str(path), str(path.parent / 'annotation.mcap')]})
    elif dataset == 'hifi_umi' and not (source / 'source').is_dir():
        shards = m.discover_shards(source)
        remaining = task_limit
        with progress('Indexing HiFi-UMI shards', len(shards), unit='shard') as bar:
            for shard in shards:
                reader = m.ShardReader(shard)
                indices = sorted(reader.episodes)[:remaining]
                if indices:
                    shared.extend(shard / 'meta' / name for name in ('info.json', 'modality.json', 'tasks.parquet'))
                    tasks.append({'source': str(shard), 'layout': 'lerobot_v3_shard',
                                  'episode_ids': [m.shard_episode_id(shard, i) for i in indices],
                                  'inputs': [str(p) for p in reader.inputs(indices)]})
                bar.update()
                if remaining is not None:
                    remaining -= len(indices)
                    if remaining == 0:
                        break
    elif dataset in ('molmoact2', 'hifi_umi'):
        meta = source / ('source/meta' if dataset == 'molmoact2' else 'source')
        info = load_json(meta / 'info.json')
        shared = [meta / 'info.json', meta / 'tasks.parquet']
        if dataset == 'molmoact2':
            m.validate_source(info)
            m.load_tasks(meta)
            shared.append(meta / 'tasks_annotated.parquet')
        else:
            m.validate_source(info, load_json(meta / 'modality.json'))
            m.load_tasks(meta / 'tasks.parquet')
            shared.append(meta / 'modality.json')
        episodes = sorted(p for p in (source / 'episodes').iterdir() if p.is_dir())
        selected = episodes if task_limit is None else episodes[:task_limit]
        with progress('Discovering episodes', len(selected), unit='episode') as bar:
            for path in selected:
                if not re.fullmatch(r'episode_\d+', path.name):
                    raise ValueError(f'Invalid episode directory: {path}')
                tasks.append({'source': str(path), 'episode_ids': [path.name],
                              'inputs': [str(path / 'data.parquet'), str(path / 'episode.json'),
                                         *(str(path / 'videos' / f'{key}.mp4') for key in m.CAMERAS)]})
                bar.update()
    elif dataset == 'galaxea':
        directories = m.discover_tasks(source)
        with progress('Discovering tasks', len(directories), unit='task') as bar:
            for directory in directories:
                if task_limit is not None and len(tasks) >= task_limit:
                    break
                bar.update()
                info = load_json(directory / 'meta/info.json')
                if config['params']['robot_type'] is not None and info.get('robot_type') != config['params']['robot_type']:
                    continue
                _, cameras = m.validate_source(info)
                metadata = m.read_jsonl(directory / 'meta/episodes.jsonl', 'episode_index')
                m.read_jsonl(directory / 'meta/tasks.jsonl', 'task_index')
                chunks = info['chunks_size']
                if len(metadata) != info['total_episodes'] or type(chunks) is not int or chunks < 1:
                    raise ValueError(f'{directory}: invalid episode metadata/chunks_size')
                shared.extend(directory / 'meta' / name for name in ('info.json', 'episodes.jsonl', 'tasks.jsonl'))
                remaining = None if task_limit is None else task_limit - len(tasks)
                for index in sorted(metadata)[:remaining]:
                    episode_id = f'{directory.name}_episode_{index:06d}'
                    paths = [m.source_path(directory, info['data_path'], index, chunks)]
                    paths += [m.source_path(directory, info['video_path'], index, chunks, key) for key in cameras]
                    tasks.append({'source': str(directory), 'index': index, 'episode_ids': [episode_id],
                                  'inputs': [str(p) for p in paths]})
    elif dataset == 'agibotworld2026':
        if source.is_dir():
            archives = []
            with progress('Scanning for archives', unit='archive') as bar:
                for path in source.rglob('*.tar.gz'):
                    archives.append(path)
                    bar.update()
            paths = sorted(archives)
        else:
            paths = [source]
        count = 0
        selected = paths if config['limit'] is None else paths[:config['limit']]
        with progress('Indexing archives', len(selected), unit='archive') as bar:
            for path in selected:
                _, _, entries = m.archive_index(path)
                tasks.append({'source': str(path), 'episode_ids': list(entries.values()), 'inputs': [str(path)]})
                count += len(entries)
                bar.update()
                if config['limit_episodes'] is not None and count >= config['limit_episodes']:
                    break
    remaining = config['limit_episodes']
    if remaining is not None:
        selected = []
        for task in tasks:
            task['episode_ids'] = task['episode_ids'][:remaining]
            selected.append(task)
            remaining -= len(task['episode_ids'])
            if remaining == 0:
                break
        tasks = selected
    if not tasks:
        raise ValueError(f'No tasks found for {dataset}: {source}')
    return tasks, sorted(set(shared))


class Adapter:
    def __init__(self, config):
        self.config = config
        self.m = module(config['dataset'])
        self.cache = {}

    def context(self, key, factory):
        if key not in self.cache:
            self.cache[key] = factory()
        return self.cache[key]

    def episode(self, task, destination):
        m, config = self.m, self.config
        dataset, source, params = config['dataset'], Path(task['source']), config['params']
        model = Path(config['model_dir']) if config['model_dir'] else None
        if dataset in ('abc130k', 'molmoact2'):
            fk = self.context('fk', lambda: m.YamFK(model, **({'arm_local': True} if dataset == 'molmoact2' else {})))
            if dataset == 'abc130k':
                return m.convert_episode(source, destination, fk)
            tasks, annotated = self.context('tasks', lambda: m.load_tasks(Path(config['source']) / 'source/meta'))
            return m.convert_episode(source, destination, tasks, annotated, fk)
        if dataset == 'hifi_umi':
            tasks = self.context('tasks', lambda: m.load_tasks(Path(config['source']) / 'source/tasks.parquet'))
            return m.convert_episode(source, destination, tasks, params['gripper_closed_rad'], params['gripper_open_rad'])
        info, metadata, tasks = self.context(str(source), lambda: (
            load_json(source / 'meta/info.json'), m.read_jsonl(source / 'meta/episodes.jsonl', 'episode_index'),
            {i: r['task'] for i, r in m.read_jsonl(source / 'meta/tasks.jsonl', 'task_index').items()}))
        kind, _ = m.validate_source(info)
        fk = self.context(kind, lambda: m.GalaxeaKinematics(model / f'galaxea_{kind}', kind, params['tcp_offsets'][kind]))
        index = task['index']
        return m.convert_episode(source, index, metadata[index], tasks, fk, info, destination)

    def process(self, task, pending, staging, store, failed):
        if self.config['dataset'] == 'agibotworld2026':
            return self.archive(task, pending, staging, store, failed)
        if self.config['dataset'] == 'hifi_umi' and task.get('layout') == 'lerobot_v3_shard':
            return self.shard(task, pending, staging, store, failed)
        episode_id = pending[0]
        destination = staging / episode_id
        record = self.episode(task, destination)
        if record is None:
            store.skip(episode_id, 'Galaxea chassis commands do not meet the stationary selection rule')
        else:
            store.commit(episode_id, destination, record)

    def shard(self, task, pending, staging, store, failed):
        source = Path(task['source'])
        reader = self.m.ShardReader(source)
        indices = {self.m.shard_episode_id(source, i): i for i in reader.episodes}
        params = self.config['params']
        for episode_id in pending:
            with tempfile.TemporaryDirectory(prefix='episode-', dir=staging) as temporary:
                destination = Path(temporary) / episode_id
                try:
                    record = reader.convert_episode(indices[episode_id], destination,
                                                    params['gripper_closed_rad'], params['gripper_open_rad'])
                    store.commit(episode_id, destination, record)
                except Exception as exc:
                    failed(episode_id, exc)

    def archive(self, task, pending, staging, store, failed):
        fk = self.context('fk', lambda: self.m.G2FK(Path(self.config['model_dir']),
                                                  self.config['params']['tcp_offset']))
        self.m.convert_archive(Path(task['source']), staging, fk, episode_ids=pending,
                               on_episode=lambda record, directory: store.commit(record['episode_id'], directory, record),
                               on_error=failed)
