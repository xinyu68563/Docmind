import json
import time
import uuid
import pytest
from docmind.server.task_store import TaskStore
from docmind.server.worker import Worker, StubEngine, TaskRecord

def wait(worker, id):
    deadline=time.monotonic()+3
    while time.monotonic()<deadline:
        rec=worker.get_task(id)
        if rec.status in {'completed','failed','cancelled'}:
            return rec.to_dict()
        time.sleep(.01)
    raise AssertionError('Task did not terminate')

def test_completed_answer_survives_restart_without_reexecuting(tmp_path):
    state=tmp_path/'state'; journals=tmp_path/'journal'
    first=Worker(StubEngine(stage_delay_ms=0),state_dir=state,journal_dir=journals)
    first.start()
    try:
        submitted,code=first.submit({'question':'保存中文问题'})
        assert code==202
        before=wait(first,submitted['task_id'])
    finally:first.shutdown()
    engine=StubEngine(stage_delay_ms=0)
    second=Worker(engine,state_dir=state,journal_dir=journals)
    second.start()
    try:
        after=second.get_task(submitted['task_id']).to_dict()
        assert after['result']==before['result'] and after['question']=='保存中文问题'
        assert after['status']=='completed' and engine.runs==0
        assert second.list_tasks()[0]['question']=='保存中文问题'
    finally:second.shutdown()

@pytest.mark.parametrize('status',['running','queued'])
def test_interrupted_tasks_are_failed_and_never_requeued(tmp_path,status):
    state=tmp_path/'state'; store=TaskStore(state,'stub'); store.acquire()
    rec=TaskRecord(uuid.uuid4().hex,'unfinished',{'question':'unfinished'},status=status)
    store.save(rec.to_dict()); store.close()
    engine=StubEngine(stage_delay_ms=0)
    worker=Worker(engine,state_dir=state,journal_dir=tmp_path/'journal');worker.start()
    try:
        restored=worker.get_task(rec.task_id)
        assert restored.status=='failed'
        assert restored.error['error_type']=='InterruptedByRestart'
        assert restored.result['usage']['output_tokens'] is None
        assert engine.runs==0 and worker.health()['queue']['size']==0
    finally:worker.shutdown()

def test_store_rejects_second_live_owner(tmp_path):
    first=TaskStore(tmp_path,'stub');second=TaskStore(tmp_path,'stub');first.acquire()
    try:
        with pytest.raises(RuntimeError):second.acquire()
    finally:first.close()
    second.acquire();second.close()

def test_atomic_failure_preserves_previous_snapshot(tmp_path,monkeypatch):
    import docmind.server.task_store as module
    store=TaskStore(tmp_path,'stub');store.acquire()
    rec=TaskRecord(uuid.uuid4().hex,'test',{'question':'test'})
    try:
        store.save(rec.to_dict());before=(tmp_path/f'{rec.task_id}.json').read_bytes()
        monkeypatch.setattr(module.os,'replace',lambda *_:(_ for _ in ()).throw(OSError('disk full')))
        rec.status='running'
        with pytest.raises(OSError):store.save(rec.to_dict())
        assert (tmp_path/f'{rec.task_id}.json').read_bytes()==before
        assert not list(tmp_path.glob('*.tmp'))
    finally:store.close()

def test_corrupted_record_is_reported_and_worker_not_ready(tmp_path):
    state=tmp_path/'state';state.mkdir();(state/'bad.json').write_text('{',encoding='utf-8')
    worker=Worker(StubEngine(),state_dir=state,journal_dir=tmp_path/'journal');worker.start()
    try:
        assert not worker.health()['ready']
        assert worker.health()['persistence']['recovery_errors']
        assert worker.submit({'question':'q'})[1]==503
    finally:worker.shutdown()

def test_new_task_not_accepted_when_persistence_fails(tmp_path,monkeypatch):
    worker=Worker(StubEngine(),state_dir=tmp_path/'state',journal_dir=tmp_path/'journal');worker.start()
    try:
        monkeypatch.setattr(worker.store,'save',lambda *_:(_ for _ in ()).throw(OSError('disk full')))
        assert worker.submit({'question':'q'})[1]==503
        assert worker.engine.runs==0
    finally:worker.shutdown()
