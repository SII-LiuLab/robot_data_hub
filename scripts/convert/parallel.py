#!/usr/bin/env python3
"""Resumable CPU conversion: prepare, statically assigned workers, then publish."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import uuid

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ('abc130k', 'molmoact2', 'hifi_umi', 'galaxea', 'agibotworld2026')


class StateError(ValueError):
    """Inconsistent recovery state: never overwrite or silently skip it."""


class Progress:
    """Dependency-free stderr progress bar, silent when stderr is not a TTY."""

    def __init__(self, label, total=None, *, unit='item'):
        self.label = label
        self.total = total
        self.unit = unit
        self.count = 0
        self.enabled = (sys.stderr.isatty()
                        and os.environ.get('ROBOT_DATA_HUB_NO_PROGRESS') != '1')
        self._rendered_at = 0.0
        self._width = 0

    def __enter__(self):
        self._render()
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def update(self, step=1):
        self.count += step
        if time.monotonic() - self._rendered_at >= 0.1:
            self._render()

    def _render(self):
        if not self.enabled:
            return
        self._rendered_at = time.monotonic()
        if self.total:
            fraction = min(self.count / self.total, 1.0)
            cells = 24
            done = int(fraction * cells)
            bar = '=' * done + ('>' if done < cells else '')
            bar += ' ' * (cells - len(bar))
            line = f'{self.label}: [{bar}] {self.count}/{self.total} ({fraction:4.0%})'
        else:
            line = f'{self.label}: {self.count} {self.unit}{"" if self.count == 1 else "s"}'
        sys.stderr.write('\r\x1b[2K' + line + ' ' * max(self._width - len(line), 0))
        sys.stderr.flush()
        self._width = len(line)

    def close(self):
        if not self.enabled:
            return
        self._render()
        sys.stderr.write('\n')
        sys.stderr.flush()
        self.enabled = False


class _NullProgress:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, step=1):
        pass


def progress(label, total=None, *, unit='item'):
    """Create a progress bar; rendering is disabled when stderr is not a TTY."""
    return Progress(label, total, unit=unit)


def _no_progress(label, total=None, *, unit='item'):
    return _NullProgress()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError) as exc:
        raise StateError(f'Cannot read {path}: {exc}') from exc


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_text(path, text):
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('x', encoding='utf-8') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_text(path, encoded(value) + '\n')


def input_stamp(path):
    path = Path(path)
    if not path.exists():
        return {'path': str(path), 'size': None, 'mtime_ns': None}
    stat = path.stat()
    if not path.is_file():
        raise ValueError(f'Expected source file: {path}')
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def check_inputs(stamps, progress=_no_progress):
    with progress('Verifying inputs', len(stamps), unit='file') as bar:
        for stamp in stamps:
            if input_stamp(stamp['path']) != stamp:
                raise ValueError(f'Source changed since prepare: {stamp["path"]}; use a new output')
            bar.update()


def validate_plan(plan):
    if not isinstance(plan, dict) or plan.get('version') != 1:
        raise StateError('Unsupported or damaged plan')
    if plan.get('id') != digest({k: v for k, v in plan.items() if k != 'id'}):
        raise StateError('Plan checksum mismatch')
    ids = []
    for task in plan['tasks']:
        if not task['episode_ids']:
            raise StateError('Empty scheduling task')
        ids.extend(task['episode_ids'])
    if (not ids or len(set(ids)) != len(ids)
            or any(not isinstance(i, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', i) for i in ids)):
        raise StateError('Episode IDs must be unique safe identifiers')


class Store:
    def __init__(self, output, plan):
        self.output = Path(output).resolve()
        self.work = self.output.with_name(self.output.name + '.work')
        self.plan = plan
        validate_plan(plan)
        if plan['output'] != str(self.output):
            raise StateError('Plan belongs to a different output')
        self.ids = [i for task in plan['tasks'] for i in task['episode_ids']]
        self.id_set = set(self.ids)
        self.dataset = self.work / 'dataset'

    @classmethod
    def load(cls, output):
        output = Path(output).resolve()
        return cls(output, read_json(output.with_name(output.name + '.work') / 'plan.json'))

    def state(self, episode_id, *, published=False):
        from scripts.convert.export_validation import file_sizes, validate_record
        if episode_id not in self.id_set:
            raise StateError(f'Unknown episode: {episode_id}')
        path = self.work / 'records' / f'{episode_id}.json'
        directory = (self.output if published else self.dataset) / 'episodes' / episode_id
        if not path.exists():
            if directory.exists() or directory.is_symlink():
                raise StateError(f'{episode_id}: episode exists without record')
            return None
        if path.is_symlink():
            raise StateError(f'{episode_id}: symlink recovery record')
        record = read_json(path)
        if (not isinstance(record, dict) or record.get('plan_id') != self.plan['id']
                or record.get('episode_id') != episode_id):
            raise StateError(f'{episode_id}: recovery record belongs to a different plan')
        try:
            if record.get('outcome') == 'skipped':
                if (self.plan['config']['dataset'] != 'galaxea'
                        or not isinstance(record.get('reason'), str) or not record['reason'].strip()
                        or directory.exists() or directory.is_symlink()):
                    raise ValueError('invalid skip record or unexpected episode directory')
                return record
            if record.get('outcome') != 'exported':
                raise ValueError('invalid record outcome')
            validate_record(record.get('manifest'), episode_id)
            if (not isinstance(record.get('files'), dict) or not record['files']
                    or any(type(n) is not int or n <= 0 for n in record['files'].values())):
                raise ValueError('invalid recorded file sizes')
            if not directory.exists() and not directory.is_symlink():
                return None
            if file_sizes(directory, record['manifest']) != record['files']:
                raise ValueError('committed files changed')
        except (ValueError, OSError) as exc:
            raise StateError(f'{episode_id}: damaged recovery state: {exc}') from exc
        return record

    def audit(self, *, published=False, complete=False, progress=_no_progress):
        records = self.work / 'records'
        if records.is_symlink():
            raise StateError('Symlink records directory')
        actual = {p.name for p in records.iterdir()} if records.exists() else set()
        if actual - {f'{i}.json' for i in self.ids}:
            raise StateError('Unexpected files in records directory')
        dataset = self.output if published else self.dataset
        episodes = dataset / 'episodes'
        if dataset.is_symlink() or episodes.is_symlink():
            raise StateError('Symlink dataset/episodes directory')
        actual = {p.name for p in episodes.iterdir()} if episodes.exists() else set()
        if actual - self.id_set:
            raise StateError('Unexpected episode directories')
        manifest, missing = [], []
        with progress('Auditing episodes', len(self.ids), unit='episode') as bar:
            for episode_id in self.ids:
                state = self.state(episode_id, published=published)
                if state is None:
                    missing.append(episode_id)
                elif state['outcome'] == 'exported':
                    manifest.append(state['manifest'])
                bar.update()
        if complete and missing:
            raise StateError(f'{len(missing)} unfinished episodes, including {missing[:3]}')
        return manifest

    def published(self, progress=_no_progress):
        if not self.output.exists():
            return False
        if self.dataset.exists():
            raise StateError('Both published output and working dataset exist')
        if {p.name for p in self.output.iterdir()} != {'episodes', 'episodes.jsonl'}:
            raise StateError('Existing output is not a published dataset')
        expected = self.audit(published=True, complete=True, progress=progress)
        try:
            actual = [json.loads(line) for line in (self.output / 'episodes.jsonl').read_text().splitlines()]
        except (ValueError, OSError) as exc:
            raise StateError('Damaged published manifest') from exc
        if actual != expected:
            raise StateError('Published manifest differs from recovery records')
        return True

    def commit(self, episode_id, directory, manifest):
        from scripts.convert.export_validation import validate_episode
        if self.state(episode_id) is not None:
            raise StateError(f'{episode_id}: already committed')
        if manifest.get('episode_id') != episode_id or directory.name != episode_id:
            raise StateError('Conversion returned a different episode ID')
        sizes = validate_episode(directory, manifest)
        # Flush data before the recovery record and directory commit become durable.
        for name in sizes:
            with (directory / name).open('rb') as handle:
                os.fsync(handle.fileno())
        for folder in (directory / 'state', directory / 'rgb', directory):
            sync_directory(folder)
        atomic_json(self.work / 'records' / f'{episode_id}.json',
                    {'plan_id': self.plan['id'], 'episode_id': episode_id,
                     'outcome': 'exported', 'manifest': manifest, 'files': sizes})
        destination = self.dataset / 'episodes' / episode_id
        if destination.exists():
            raise StateError(f'{episode_id}: destination appeared during conversion')
        directory.rename(destination)
        sync_directory(destination.parent)
        sync_directory(directory.parent)
        (self.work / 'errors' / f'{episode_id}.err').unlink(missing_ok=True)
        print(f'Committed {episode_id}', flush=True)

    def skip(self, episode_id, reason):
        if self.plan['config']['dataset'] != 'galaxea':
            raise StateError('Only Galaxea has a legitimate skip policy')
        if self.state(episode_id) is not None:
            raise StateError(f'{episode_id}: already completed')
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError('A skip needs an explicit reason')
        atomic_json(self.work / 'records' / f'{episode_id}.json',
                    {'plan_id': self.plan['id'], 'episode_id': episode_id,
                     'outcome': 'skipped', 'reason': reason})
        (self.work / 'errors' / f'{episode_id}.err').unlink(missing_ok=True)
        print(f'Skipped {episode_id}: {reason}', flush=True)

    def failure(self, episode_id, source, exc):
        message = f'{episode_id}\nInput: {source}\n' + ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        print(message, file=sys.stderr, flush=True)
        atomic_text(self.work / 'errors' / f'{episode_id}.err', message)


def require_single_process():
    if int(os.environ.get('SLURM_NTASKS', '1')) > 1 and 'SLURM_PROCID' in os.environ:
        raise ValueError('prepare, finalize and local run require one process; launch worker with srun')


def prepare(config, output):
    require_single_process()
    from scripts.convert.parallel_adapters import discover
    output = Path(output).resolve()
    work = output.with_name(output.name + '.work')
    source = Path(config['source'])
    if output == source or output.is_relative_to(source) or work.is_relative_to(source):
        raise ValueError('Output/work directory must be outside the input source')
    model_dir = config.get('model_dir')
    if model_dir and not Path(model_dir).is_dir():
        raise ValueError(f'Model directory does not exist: {model_dir}')
    if (work / 'plan.json').exists():
        store = Store.load(output)
        if store.plan['config'] != config:
            raise ValueError('Incompatible conversion config; use a new output')
    else:
        if output.exists():
            raise StateError('Output already exists without a matching resume plan')
        if work.exists() and any(p.name not in ('plan.json', 'logs') and not p.name.endswith('.tmp') for p in work.iterdir()):
            raise StateError('Work directory has data but no plan')
        tasks, shared = discover(config, progress)
        plan = {'version': 1, 'config': config, 'output': str(output),
                'tasks': tasks, 'shared_inputs': [input_stamp(p) for p in shared]}
        for task in tasks:
            task['inputs'] = [input_stamp(p) for p in task['inputs']]
        plan['id'] = digest(plan)
        store = Store(output, plan)
        work.mkdir(parents=True, exist_ok=True)
        atomic_json(work / 'plan.json', plan)
    stamps = list(store.plan['shared_inputs'])
    for task in store.plan['tasks']:
        stamps += task['inputs']
    check_inputs(stamps, progress)
    if store.published(progress):
        print(f'Already complete: {output}', flush=True)
        return store
    for name in ('records', 'errors', 'logs', 'staging', 'dataset/episodes'):
        directory = work / name
        if directory.is_symlink():
            raise StateError(f'Symlink working directory: {directory}')
        directory.mkdir(parents=True, exist_ok=True)
    # Only prepare may clean attempts and diagnostics, and only after previous workers have exited.
    for name in ('logs', 'staging'):
        for path in (work / name).iterdir():
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
    for directory in (work, work / 'records', work / 'errors', store.dataset):
        for path in directory.glob('.*.tmp'):
            path.unlink()
    store.audit(progress=progress)
    print(f'Prepared {len(store.plan["tasks"])} tasks / {len(store.ids)} episodes: {work}', flush=True)
    return store


def configure_worker():
    # Set before importing numpy/Arrow/codec modules in child processes.
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[key] = '1'
    os.environ['ROBOT_DATA_HUB_SERIAL_CODECS'] = '1'
    import pyarrow as pa
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)


@contextmanager
def worker_diagnostics(output):
    """Keep CLI startup failures and native crash stacks outside episode records."""
    output = Path(output).resolve()
    directory = output.with_name(output.name + '.work') / 'logs'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'worker-{uuid.uuid4().hex}.err'
    with path.open('x', encoding='utf-8', buffering=1) as handle:
        def report(message):
            print(f'{time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())} {message}',
                  file=handle, flush=True)

        report(f'Start host={socket.gethostname()} pid={os.getpid()} output={output}')
        report(f'Python: {sys.executable}; argv={sys.argv!r}')
        for key in ('SLURM_JOB_ID', 'SLURM_STEP_ID', 'SLURM_PROCID', 'SLURM_NTASKS'):
            report(f'{key}={os.environ.get(key, "<unset>")}')
        print(f'Worker diagnostics: {path}', file=sys.stderr, flush=True)
        was_enabled = faulthandler.is_enabled()
        faulthandler.enable(file=handle)
        try:
            yield report
        except BaseException:
            report('Worker terminated by exception:')
            traceback.print_exc(file=handle)
            handle.flush()
            raise
        finally:
            faulthandler.disable()
            if was_enabled:
                faulthandler.enable()


def worker(output, worker_id, worker_count, *, report=None):
    report = report or (lambda message: None)
    report(f'Worker {worker_id}/{worker_count}: configuring dependencies')
    if worker_count < 1 or not 0 <= worker_id < worker_count:
        raise ValueError('Require worker_count > 0 and 0 <= worker_id < worker_count')
    configure_worker()
    from scripts.convert.parallel_adapters import Adapter
    report('Loading plan')
    store = Store.load(output)
    report('Checking shared inputs and published output')
    check_inputs(store.plan['shared_inputs'])
    if store.published():
        return 0
    report('Initializing adapter')
    adapter = Adapter(store.plan['config'])
    failures = 0
    for index, task in enumerate(store.plan['tasks']):
        if index % worker_count != worker_id:
            continue
        report(f'Task {index}: input={task["source"]}; episodes={task["episode_ids"]}')
        check_inputs(task['inputs'])
        pending = [i for i in task['episode_ids'] if store.state(i) is None]
        if not pending:
            continue
        with tempfile.TemporaryDirectory(prefix=f'w{worker_id}-', dir=store.work / 'staging') as temporary:
            def failed(episode_id, exc):
                nonlocal failures
                if isinstance(exc, StateError):
                    raise exc
                failures += 1
                report(f'Episode {episode_id} failed: {type(exc).__name__}: {exc}')
                store.failure(episode_id, task['source'], exc)
            try:
                adapter.process(task, pending, Path(temporary), store, failed)
            except StateError:
                raise
            except Exception as exc:
                # Archive-level I/O errors affect only episodes not yet committed.
                for episode_id in pending:
                    if store.state(episode_id) is None:
                        failed(episode_id, exc)
    return 1 if failures else 0


def finalize(output):
    require_single_process()
    store = Store.load(output)
    if store.published():
        return
    manifest = store.audit(complete=True)
    atomic_text(store.dataset / 'episodes.jsonl', ''.join(encoded(r) + '\n' for r in manifest))
    if store.output.exists():
        raise StateError('Output appeared before publication')
    store.dataset.rename(store.output)
    sync_directory(store.output.parent)
    sync_directory(store.work)
    print(f'Published {len(manifest)} episodes: {store.output}', flush=True)


def run(config, output, workers=1):
    """Local convenience runner. Cluster scheduling belongs to the caller."""
    require_single_process()
    if type(workers) is not int or workers < 1:
        raise ValueError('workers must be a positive integer')
    configure_worker()
    store = prepare(config, output)
    if store.published():
        return
    command = [sys.executable, '-m', 'scripts.convert.parallel', 'worker', '--output-dir', str(store.output)]
    processes = []
    try:
        for rank in range(workers):
            processes.append(subprocess.Popen([*command, '--worker-id', str(rank), '--worker-count', str(workers)], cwd=ROOT))
        statuses = [p.wait() for p in processes]
        if any(statuses):
            raise RuntimeError(f'Workers failed ({statuses}); rerun the same command to resume')
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
    finalize(store.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name, help_text in (
            ('prepare', 'create or validate the plan and clean interrupted attempts and diagnostics; one process'),
            ('run', 'local only: prepare, launch local workers, finalize')):
        launch = sub.add_parser(name, help=help_text)
        launch.add_argument('--dataset', choices=DATASETS, required=True)
        launch.add_argument('--input-dir', type=Path, required=True)
        launch.add_argument('--output-dir', type=Path, required=True)
        launch.add_argument('--model-dir', type=Path, help='model directory; Galaxea expects the parent of its two model packages')
        if name == 'run':
            launch.add_argument('--workers', type=int, default=1, help='local worker count (default 1)')
        launch.add_argument('--limit', type=int, help='first N scheduling tasks (archives for AgiBot, episodes otherwise)')
        launch.add_argument('--limit-episodes', type=int, help='first N candidate episodes globally, including legitimate skips')
        launch.add_argument('--robot-type', choices=('r1lite', 'r1pro'), help='Galaxea embodiment filter')
        launch.add_argument('--tcp-offset', type=float, nargs=3, help='AgiBot TCP offset in metres')
        for kind in ('r1lite', 'r1pro'):
            launch.add_argument(f'--{kind}-tcp-offset', type=float, nargs=3, help='Galaxea TCP offset in metres')
        launch.add_argument('--gripper-open-rad', type=float, help='HiFi-UMI open angle')
        launch.add_argument('--gripper-closed-rad', type=float, help='HiFi-UMI closed angle')
    child = sub.add_parser('worker', help='process assigned tasks; caller launches with srun after prepare')
    child.add_argument('--output-dir', type=Path, required=True)
    child.add_argument('--worker-id', type=int, default=None)
    child.add_argument('--worker-count', type=int, default=None)
    finish = sub.add_parser('finalize', help='verify all tasks and publish; one process after workers exit')
    finish.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'worker':
            with worker_diagnostics(args.output_dir) as report:
                rank = args.worker_id if args.worker_id is not None else int(os.environ['SLURM_PROCID'])
                count = args.worker_count if args.worker_count is not None else int(os.environ['SLURM_NTASKS'])
                status = worker(args.output_dir, rank, count, report=report)
                report(f'Worker exit code: {status}; episode errors: {args.output_dir}.work/errors/')
                return status
        if args.command == 'finalize':
            finalize(args.output_dir)
            return 0
        if args.command == 'run' and args.workers < 1:
            raise ValueError('--workers must be positive')
        for name in ('limit', 'limit_episodes'):
            if getattr(args, name) is not None and getattr(args, name) < 1:
                raise ValueError(f'--{name.replace("_", "-")} must be positive')
        configure_worker()
        from scripts.convert.parallel_adapters import configuration
        if args.command == 'prepare':
            prepare(configuration(args), args.output_dir)
        else:
            run(configuration(args), args.output_dir, args.workers)
    except KeyboardInterrupt:
        print('Interrupted; wait for all workers to exit, then rerun to resume.', file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(f'Error: {exc}', file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, interrupted)
    sys.exit(main())
