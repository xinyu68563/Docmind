package com.chenxinyu.docmind.worker;

import java.util.Map;

/**
 * worker 侧错误的统一抽象。
 *
 * 为什么每种失败都要有自己的类型：
 *   "队列满了"（429，客户端稍后重试即可）和"worker 挂了"（503，需要人来处理）
 *   是完全不同的两件事。如果都变成一句笼统的"后端错误 500"，
 *   调用方就只能靠猜来决定要不要重试 —— 而猜错的代价是雪崩式重试。
 */
public class WorkerException extends RuntimeException {

    private final int httpStatus;
    private final Map<String, Object> body;

    protected WorkerException(int httpStatus, String message, Map<String, Object> body) {
        super(message);
        this.httpStatus = httpStatus;
        this.body = body == null ? Map.of() : body;
    }

    public int getHttpStatus() {
        return httpStatus;
    }

    public Map<String, Object> getBody() {
        return body;
    }

    /** worker 不可达、超时或明确报 503：不重试也不伪造，如实上报。 */
    public static class Unavailable extends WorkerException {
        public Unavailable(String message, Map<String, Object> body) {
            super(503, message, body);
        }
    }

    /** 有界队列已满：明确的背压信号，客户端应当稍后重试。 */
    public static class QueueFull extends WorkerException {
        public QueueFull(String message, Map<String, Object> body) {
            super(429, message, body);
        }
    }

    /** worker 认为请求体不合法（我们的校验和它的校验可能不完全一致）。 */
    public static class Rejected extends WorkerException {
        public Rejected(String message, Map<String, Object> body) {
            super(400, message, body);
        }
    }
    public static class SourceIntegrity extends WorkerException {
        public SourceIntegrity(String message,Map<String,Object> body){super(409,message,body);}
    }

    /** worker 上不存在这个 task_id。 */
    public static class TaskNotFound extends WorkerException {
        public TaskNotFound(String message, Map<String, Object> body) {
            super(404, message, body);
        }
    }

    /** worker 返回了未预期的状态码。 */
    public static class Unexpected extends WorkerException {
        public Unexpected(int status, String message, Map<String, Object> body) {
            super(status == 0 ? 502 : status, message, body);
        }
    }
}
