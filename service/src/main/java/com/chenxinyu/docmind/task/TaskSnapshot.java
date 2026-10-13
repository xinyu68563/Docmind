package com.chenxinyu.docmind.task;

import java.time.Instant;
import java.util.List;
import java.util.Map;

/**
 * 一个任务在服务侧的最新已知状态。
 *
 * 【这不是持久层】
 *   它只是一份内存缓存，用来在 worker 短暂不可达时仍能回答"上次看到它是什么状态"。
 *   服务重启后所有任务状态丢失 —— 这是刻意的最小实现，
 *   README 和 API 里都必须写明，绝不能让用户以为任务被持久化了。
 *   真正的留痕（逐调用日志）由 Python worker 写在 logs/service/worker 下，
 *   那个是持久的，和这里的内存状态是两回事。
 */
public class TaskSnapshot {

    private final String taskId;
    private volatile String status;
    private volatile String stage;
    private volatile String question;
    public String getQuestion(){return question;}
    private volatile List<String> documentIds=List.of();
    public List<String> getDocumentIds(){return documentIds;}
    private volatile String documentId;
    public String getDocumentId(){return documentId;}
    private volatile Map<String, Object> result;
    private volatile Map<String, Object> error;
    private volatile List<Map<String, Object>> events = List.of();
    private volatile String backend;
    private volatile boolean workerReachable = true;
    private volatile Instant updatedAt = Instant.now();

    public TaskSnapshot(String taskId, String status) {
        this.taskId = taskId;
        this.status = status;
        this.stage = status;
    }

    public String getTaskId() {
        return taskId;
    }

    public String getStatus() {
        return status;
    }

    public String getStage() {
        return stage;
    }

    public Map<String, Object> getResult() {
        return result;
    }

    public Map<String, Object> getError() {
        return error;
    }

    public List<Map<String, Object>> getEvents() {
        return events;
    }

    public String getBackend() {
        return backend;
    }

    public boolean isWorkerReachable() {
        return workerReachable;
    }

    public Instant getUpdatedAt() {
        return updatedAt;
    }

    @SuppressWarnings("unchecked")
    public void updateFrom(Map<String, Object> workerTask, boolean reachable) {
        this.workerReachable = reachable;
        if (!reachable) {
            this.updatedAt = Instant.now();
            return;
        }
        this.question = (String) workerTask.getOrDefault("question", this.question);
        if(workerTask.get("params") instanceof Map<?,?> params) {
            this.documentId=(String)params.get("document_id");
            this.documentIds=params.get("document_ids") instanceof List<?> ids?(List<String>)ids:(documentId==null?List.of():List.of(documentId));
        }
        this.status = (String) workerTask.getOrDefault("status", this.status);
        this.stage = (String) workerTask.getOrDefault("stage", this.stage);
        this.result = (Map<String, Object>) workerTask.get("result");
        this.error = (Map<String, Object>) workerTask.get("error");
        Object ev = workerTask.get("events");
        if (ev instanceof List<?> list) {
            this.events = list.stream().filter(Map.class::isInstance)
                    .map(e -> (Map<String, Object>) e).toList();
        }
        if (this.result != null) {
            Object b = this.result.get("backend");
            if (b != null) {
                this.backend = String.valueOf(b);
            }
        }
        this.updatedAt = Instant.now();
    }

    public void markUnreachable() {
        this.workerReachable = false;
        this.updatedAt = Instant.now();
    }

    public boolean isTerminal() {
        return "completed".equals(status) || "failed".equals(status) || "cancelled".equals(status);
    }
}
