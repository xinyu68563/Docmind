package com.chenxinyu.docmind.task;

import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import com.chenxinyu.docmind.config.DocMindProperties;
import com.chenxinyu.docmind.worker.WorkerException;
import com.chenxinyu.docmind.worker.WorkerClient;
import jakarta.annotation.PreDestroy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

/**
 * 任务编排与 SSE 事件转发。
 *
 * 【状态来源只有一个】
 *   Java 不自己推断任务状态，一律向 worker 拉取。缓存（TaskSnapshot）
 *   只在 worker 不可达时用于回答"上次看到的是什么"。
 *   这是为了避免最常见的分布式错觉：两边各算一份状态，然后 slowly drift。
 *
 * 【SSE 只转发真实事件】
 *   worker 的每个阶段切换都会写一条带 seq 的事件；本服务只把"还没发过的"
 *   转发出去。所以 SSE 里出现的 stage 一定是真实发生过的阶段，
 *   不是前端mock、也不是定时器凑出来的心跳。
 *
 * 【客户端断开时不做什么】
 *   不取消任务。观察者和任务是两回事：一个浏览器关掉页面，
 *   不应该把别人提交的推理任务杀掉。断开只清掉订阅和轮询任务。
 */
@Service
public class TaskService {

    private static final Logger log = LoggerFactory.getLogger(TaskService.class);
    private static final int MAX_CONSECUTIVE_FAILURES = 5;

    private final WorkerClient worker;
    private final DocMindProperties props;

    private final ConcurrentHashMap<String, TaskSnapshot> snapshots = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, Listener> listeners = new ConcurrentHashMap<>();
    private final ScheduledExecutorService scheduler =
            Executors.newScheduledThreadPool(2, r -> {
                Thread t = new Thread(r, "docmind-sse-poll");
                t.setDaemon(true);
                return t;
            });

    public TaskService(WorkerClient worker, DocMindProperties props) {
        this.worker = worker;
        this.props = props;
    }

    private static final class Listener {
        final String taskId;
        final SseEmitter emitter;
        long lastSeq;
        int failures;
        volatile ScheduledFuture<?> future;
        volatile boolean closed;
        Listener(String taskId, SseEmitter emitter, long lastSeq) {
            this.taskId=taskId; this.emitter=emitter; this.lastSeq=lastSeq;
        }
    }

    // ------------------------------------------------------------------
    // 提交
    // ------------------------------------------------------------------

    public Map<String, Object> submit(Map<String, Object> payload) {
        Map<String, Object> created = worker.submit(payload);
        String taskId = String.valueOf(created.get("task_id"));
        snapshots.put(taskId, new TaskSnapshot(taskId, "queued"));
        return created;
    }

    // ------------------------------------------------------------------
    // 状态 / 结果
    // ------------------------------------------------------------------

    public Map<String, Object> status(String taskId) {
        try {
            Map<String, Object> t = worker.getTask(taskId);
            TaskSnapshot snap = snapshots.computeIfAbsent(taskId, id -> new TaskSnapshot(id, "queued"));
            snap.updateFrom(t, true);
            return view(snap, false);
        } catch (WorkerException.TaskNotFound e) {
            throw e;
        } catch (WorkerException.Unavailable e) {
            // worker 暂时不可达，但我们有上次看到的状态 ——
            // 明确标注 stale，绝不把它当成"刚刚查到的"。
            TaskSnapshot snap = snapshots.get(taskId);
            if (snap == null) {
                throw e;
            }
            snap.markUnreachable();
            return view(snap, true);
        }
    }

    public Map<String, Object> cancel(String taskId) {
        Map<String, Object> resp = worker.cancel(taskId);
        TaskSnapshot snap = snapshots.get(taskId);
        if (snap != null) {
            // 排队中的任务 worker 会直接置 cancelled；运行中的仍要等它协作停止。
            Object st = resp.get("status");
            if (st != null) {
                snap.updateFrom(Map.of("status", st, "stage",
                        "cancelled".equals(st) ? "cancelled_while_queued" : "cancel_requested"), true);
            }
        }
        return resp;
    }

    public Map<String, Object> view(TaskSnapshot snap, boolean stale) {
        Map<String, Object> out = new java.util.LinkedHashMap<>();
        out.put("task_id", snap.getTaskId());
        out.put("question", snap.getQuestion());
        out.put("document_id", snap.getDocumentId());
        out.put("document_ids",snap.getDocumentIds());
        out.put("status", snap.getStatus());
        out.put("stage", snap.getStage());
        out.put("worker_reachable", snap.isWorkerReachable());
        out.put("stale", stale);
        out.put("backend", snap.getBackend());
        out.put("stub", "stub".equals(snap.getBackend()));
        out.put("updated_at", snap.getUpdatedAt().toString());
        out.put("events", snap.getEvents());
        if (snap.getResult() != null) {
            out.put("result", snap.getResult());
        }
        if (snap.getError() != null) {
            out.put("error", snap.getError());
        }
        if (stale) {
            out.put("note", "Python worker 当前不可达，以上为最后一次观测到的状态，可能已过期");
        }
        return out;
    }

    // ------------------------------------------------------------------
    // SSE
    // ------------------------------------------------------------------

    /**
     * 订阅任务事件。
     *
     * @param lastEventId 客户端重连时带来的 Last-Event-ID；本服务会重放它之后的事件。
     *
     * 重连能力范围（README 里也写了同样的边界，不许含糊）：
     *   - 支持：重连时带上 Last-Event-ID 可补齐缓冲窗口内（默认 200 条）的事件；
     *           不带 ID 重连会先收到一条 snapshot，因此**总能拿到当前终态**。
     *   - 不支持：超出缓冲窗口的旧事件、跨服务重启的事件（内存缓冲，不落盘）。
     */
    public SseEmitter subscribe(String taskId, String lastEventId) {
        // Validate before returning an emitter; unknown tasks are actual HTTP 404.
        Map<String,Object> initial=worker.getTask(taskId);
        SseEmitter emitter=new SseEmitter(props.sse().timeoutMs());
        long cursor=0;
        try { if(lastEventId!=null)cursor=Math.max(0,Long.parseLong(lastEventId)); }
        catch(NumberFormatException ignored) {}
        String id=java.util.UUID.randomUUID().toString();
        Listener listener=new Listener(taskId,emitter,cursor);
        listeners.put(id,listener);
        emitter.onCompletion(()->release(id));
        emitter.onTimeout(()->{release(id);emitter.complete();});
        emitter.onError(e->release(id));
        // A current snapshot is always sent, even if a reconnect cursor is beyond the buffer.
        try { emitter.send(SseEmitter.event().name("status").data(initial)); }
        catch(IOException|IllegalStateException e){release(id);return emitter;}
        listener.future=scheduler.scheduleWithFixedDelay(()->poll(id,listener),
                props.sse().pollIntervalMs(),props.sse().pollIntervalMs(),TimeUnit.MILLISECONDS);
        if(listener.closed)listener.future.cancel(false);
        return emitter;
    }

    private void poll(String id, Listener listener) {
        if(listener.closed)return;
        try {
            Map<String,Object> t=worker.getTask(listener.taskId);
            TaskSnapshot snap=snapshots.computeIfAbsent(listener.taskId,k->new TaskSnapshot(k,"queued"));
            snap.updateFrom(t,true);
            listener.failures=0;
            for(Map<String,Object> ev:snap.getEvents()) {
                long seq=ev.get("seq") instanceof Number n?n.longValue():0;
                if(seq<=listener.lastSeq)continue;
                var data=new java.util.LinkedHashMap<String,Object>(ev);
                data.put("task_id",listener.taskId);
                listener.emitter.send(SseEmitter.event().id(String.valueOf(seq))
                        .name(String.valueOf(ev.getOrDefault("type","stage"))).data(data));
                listener.lastSeq=seq;
            }
            if(snap.isTerminal()) {
                listener.emitter.send(SseEmitter.event().id(String.valueOf(listener.lastSeq+1))
                        .name("terminal").data(view(snap,false)));
                release(id);listener.emitter.complete();
            }
        } catch(WorkerException.Unavailable e) {
            if(++listener.failures>=MAX_CONSECUTIVE_FAILURES) {
                terminal(id,listener,"worker_lost",e.getMessage());
            }
        } catch(WorkerException.TaskNotFound e) {
            terminal(id,listener,"not_found",e.getMessage());
        } catch(IOException|IllegalStateException e) {
            release(id);listener.emitter.complete();
        } catch(RuntimeException e) {
            terminal(id,listener,"error",String.valueOf(e.getMessage()));
        }
    }
    private void terminal(String id,Listener listener,String status,String message) {
        try {listener.emitter.send(SseEmitter.event().name("terminal")
                .data(Map.of("task_id",listener.taskId,"status",status,"message",message)));}
        catch(IOException|IllegalStateException ignored) {}
        release(id);listener.emitter.complete();
    }
    private void release(String id) {
        Listener listener=listeners.remove(id);
        if(listener!=null) {
            listener.closed=true;
            if(listener.future!=null)listener.future.cancel(false);
        }
    }
    public int activeSubscriptions(){return listeners.size();}
    public int knownTasks(){return snapshots.size();}
    @PreDestroy
    public void shutdown(){
        listeners.forEach((id,l)->{release(id);l.emitter.complete();});
        scheduler.shutdownNow();
    }
}
