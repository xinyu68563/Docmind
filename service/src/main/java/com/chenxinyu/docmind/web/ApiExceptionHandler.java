package com.chenxinyu.docmind.web;

import java.util.LinkedHashMap;
import java.util.Map;

import com.chenxinyu.docmind.worker.WorkerException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

/**
 * 把 worker 侧异常翻译成 HTTP 状态码。
 *
 * 为什么要有这一层而不是在 Controller 里 try/catch：
 *   翻译规则集中一处，才能保证"队列满 = 429"这件事在任何接口上都一致。
 *   散在各处迟早会出现某个接口把 429 写成 500 的情况。
 *
 * 为什么失败响应里一定要带 message：
 *   503 只说"服务不可用"是没法排障的。调用方至少要能区分
 *   "worker 没启动"和"worker 加载模型失败"。
 */
@RestControllerAdvice
public class ApiExceptionHandler {

    private static final Logger log = LoggerFactory.getLogger(ApiExceptionHandler.class);

    @ExceptionHandler(WorkerException.QueueFull.class)
    public ResponseEntity<Map<String, Object>> queueFull(WorkerException.QueueFull e) {
        return body(HttpStatus.TOO_MANY_REQUESTS, "queue_full", e.getMessage(), e.getBody());
    }

    @ExceptionHandler(WorkerException.Unavailable.class)
    public ResponseEntity<Map<String, Object>> unavailable(WorkerException.Unavailable e) {
        log.warn("worker 不可用: {}", e.getMessage());
        return body(HttpStatus.SERVICE_UNAVAILABLE, "worker_unavailable", e.getMessage(), e.getBody());
    }

    @ExceptionHandler(WorkerException.TaskNotFound.class)
    public ResponseEntity<Map<String, Object>> notFound(WorkerException.TaskNotFound e) {
        return body(HttpStatus.NOT_FOUND, "task_not_found", e.getMessage(), e.getBody());
    }

    @ExceptionHandler(WorkerException.Rejected.class)
    public ResponseEntity<Map<String, Object>> rejected(WorkerException.Rejected e) {
        return body(HttpStatus.BAD_REQUEST, "invalid_request", e.getMessage(), e.getBody());
    }
    @ExceptionHandler(WorkerException.SourceIntegrity.class)
    public ResponseEntity<Map<String,Object>> sourceIntegrity(WorkerException.SourceIntegrity e){
        return body(HttpStatus.CONFLICT,"source_integrity_error",e.getMessage(),e.getBody());
    }

    @ExceptionHandler(WorkerException.class)
    public ResponseEntity<Map<String, Object>> other(WorkerException e) {
        return body(HttpStatus.BAD_GATEWAY, "worker_error", e.getMessage(), e.getBody());
    }

    private ResponseEntity<Map<String, Object>> body(HttpStatus status, String error,
                                                     String message, Map<String, Object> detail) {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("error", error);
        out.put("message", message);
        if (detail != null && !detail.isEmpty()) {
            out.put("detail", detail);
        }
        return ResponseEntity.status(status).body(out);
    }
}
