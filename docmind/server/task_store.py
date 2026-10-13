"""Atomic local task snapshots. A directory has one live worker owner."""
import json
import os
import re
import threading
from pathlib import Path

SCHEMA = 'docmind.task/1'
ID = re.compile(r'^[a-f0-9]{32}$')

class TaskStore:
    def __init__(self, directory, backend):
        self.directory = Path(directory)
        self.backend = backend
        self._lock = threading.Lock()
        self._owner = None

    def acquire(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        fh = (self.directory/'.owner.lock').open('a+b')
        if fh.tell() == 0:
            fh.write(b'0'); fh.flush()
        fh.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            raise RuntimeError('Task state directory already owned by another worker')
        self._owner = fh

    def close(self):
        if self._owner:
            self._owner.close(); self._owner = None

    def save(self, snapshot):
        task_id = snapshot['task_id']
        if not ID.fullmatch(task_id):
            raise ValueError('Invalid task identifier')
        payload = {'schema': SCHEMA, 'backend': self.backend, 'task': snapshot}
        with self._lock:
            target = self.directory/f'{task_id}.json'
            temp = target.with_suffix('.tmp')
            try:
                with temp.open('w', encoding='utf-8', newline='\n') as fh:
                    json.dump(payload, fh, ensure_ascii=False)
                    fh.flush(); os.fsync(fh.fileno())
                os.replace(temp, target)
            finally:
                temp.unlink(missing_ok=True)

    def discard(self, task_id):
        if not ID.fullmatch(task_id):
            raise ValueError("Invalid task identifier")
        (self.directory/f"{task_id}.json").unlink(missing_ok=True)

    def load(self):
        tasks, errors = [], []
        for path in sorted(self.directory.glob('*.json')):
            try:
                payload = json.loads(path.read_text(encoding='utf-8'))
                task = payload['task']
                if payload.get('schema') != SCHEMA or payload.get('backend') != self.backend:
                    raise ValueError('Snapshot schema or backend mismatch')
                if not ID.fullmatch(path.stem) or task.get('task_id') != path.stem:
                    raise ValueError('Snapshot task identity mismatch')
                if task.get('status') not in {'queued','running','completed','failed','cancelled'}:
                    raise ValueError('Invalid task status')
                if not isinstance(task.get('question'), str) or not isinstance(task.get('params'), dict):
                    raise ValueError('Missing task request')
                if not isinstance(task.get('events'), list) or not all(isinstance(e, dict) for e in task['events']):
                    raise ValueError('Invalid event records')
                if task.get("result") is not None and not isinstance(task["result"], dict):
                    raise ValueError("Invalid task result")
                if task["status"] == "completed" and not isinstance(task.get("result"), dict):
                    raise ValueError("Completed task has no result")
                if not isinstance(task.get("created_at"), str):
                    raise ValueError("Invalid task time")
                tasks.append(task)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                errors.append(f'{path.name}: {type(exc).__name__}: {exc}')
        return tasks, errors
