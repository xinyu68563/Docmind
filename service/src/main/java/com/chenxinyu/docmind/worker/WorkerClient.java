package com.chenxinyu.docmind.worker;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Map;

import com.chenxinyu.docmind.config.DocMindProperties;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PreDestroy;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

/**
 * Python worker 的 HTTP 客户端。
 *
 * 【为什么用 JDK 自带的 java.net.http.HttpClient】
 *   它从 JDK 11 起就在标准库里，支持超时、异步和 HTTP/2。
 *   为一个本地回环调用引入第三方 HTTP 库，只会增加构建体积和依赖风险。
 *
 * 【为什么每个请求都带超时】
 *   没有超时的客户端等于把整个线程池的命运交给对面进程。
 *   worker 一旦卡在模型加载上，请求会一个个挂死，最终整个服务不可用 ——
 *   而监控上看到的是"服务还在，只是很慢"，极难定位。
 */
@Component
public class WorkerClient {

    private static final Logger log = LoggerFactory.getLogger(WorkerClient.class);

    private final HttpClient http;
    private final ObjectMapper mapper;
    private final String baseUrl;

    public WorkerClient(DocMindProperties props, ObjectMapper mapper) {
        this.mapper = mapper;
        this.baseUrl = props.worker().baseUrl();
        this.http = HttpClient.newBuilder()
                .connectTimeout(Duration.ofMillis(props.worker().connectTimeoutMs()))
                .build();
        this.requestTimeout = Duration.ofMillis(props.worker().requestTimeoutMs());
    }

    private final Duration requestTimeout;

    // ------------------------------------------------------------------
    // 健康检查
    // ------------------------------------------------------------------

    public Map<String, Object> health() {
        return get("/v1/health");
    }

    // ------------------------------------------------------------------
    // 任务
    // ------------------------------------------------------------------

    /** 提交任务。返回 worker 的原始响应体（含 task_id）。 */
    public Map<String, Object> submit(Map<String, Object> payload) {
        return post("/v1/tasks", payload, 202);
    }

    public Map<String, Object> getTask(String taskId) {
        return get("/v1/tasks/" + taskId);
    }

    /**
     * 取消任务。worker 对"排队中"和"运行中"返回不同状态码（200 / 202），
     * 两者都是成功，不能因为不是 200 就当成失败。
     */
    public Map<String, Object> cancel(String taskId) {
        try {
            return post("/v1/tasks/" + taskId + "/cancel", Map.of(), 200, 202);
        } catch (WorkerException.Unexpected e) {
            if (e.getHttpStatus() == 409) {
                throw new WorkerException.Unexpected(409, e.getMessage(), e.getBody());
            }
            throw e;
        }
    }

    // ------------------------------------------------------------------
    // 底层
    // ------------------------------------------------------------------

    public Map<String,Object> tasks(){ return get("/v1/tasks"); }
    public Map<String,Object> documents(){ return get("/v1/documents"); }
    public record PageImage(byte[] body, int pages){}
    public PageImage page(String id,int page){
        var req=HttpRequest.newBuilder(URI.create(baseUrl+"/v1/documents/"+encode(id)+"/pages/"+page)).timeout(requestTimeout).GET().build();
        try{
            var r=http.send(req,HttpResponse.BodyHandlers.ofByteArray());
            if(r.statusCode()==409)throw new WorkerException.SourceIntegrity("论文原文或索引已改变，请恢复原文件或重新导入",null);
            if(r.statusCode()==404)throw new WorkerException.TaskNotFound("Document not found",null);
            if(r.statusCode()==400)throw new WorkerException.Rejected("页码超出范围或PDF无法读取",null);
            if(r.statusCode()!=200||!r.headers().firstValue("Content-Type").orElse("").startsWith("image/png"))throw new WorkerException.Unavailable("Invalid page response",null);
            return new PageImage(r.body(),Integer.parseInt(r.headers().firstValue("X-PDF-Pages").orElse("0")));
        }catch(IOException|NumberFormatException e){throw new WorkerException.Unavailable("Page preview unavailable",null);}
        catch(InterruptedException e){Thread.currentThread().interrupt();throw new WorkerException.Unavailable("Interrupted",null);}
    }
    public Map<String,Object> importStatus(String id){return get("/v1/imports/"+encode(id));}
    public Map<String,Object> upload(byte[] bytes, String name){
        var req=HttpRequest.newBuilder(URI.create(baseUrl+"/v1/imports"))
                .timeout(requestTimeout).header("Content-Type","application/pdf")
                .header("X-File-Name",encode(name)).POST(HttpRequest.BodyPublishers.ofByteArray(bytes)).build();
        return send(req,200,202);
    }
    public Map<String,Object> chunk(String id){ return get("/v1/chunks/"+encode(id)); }
    private String encode(String id) { return java.net.URLEncoder.encode(id, StandardCharsets.UTF_8).replace("+","%20"); }
    public byte[] pdf(String id) {
        var req=HttpRequest.newBuilder(URI.create(baseUrl+"/v1/documents/"+encode(id)+"/pdf"))
                .timeout(requestTimeout).GET().build();
        try {
            var r=http.send(req,HttpResponse.BodyHandlers.ofByteArray());
            if(r.statusCode()==409)throw new WorkerException.SourceIntegrity("论文原文或索引已改变，请恢复原文件或重新导入",null);
            if(r.statusCode()==404)throw new WorkerException.TaskNotFound("Document not found",null);
            if(r.statusCode()!=200 || !r.headers().firstValue("Content-Type").orElse("").startsWith("application/pdf"))
                throw new WorkerException.Unavailable("Invalid document response",null);
            return r.body();
        } catch(IOException e){throw new WorkerException.Unavailable("Worker unavailable",null);}
        catch(InterruptedException e){Thread.currentThread().interrupt();throw new WorkerException.Unavailable("Interrupted",null);}
    }

    private Map<String, Object> get(String path) {
        HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + path))
                .timeout(requestTimeout)
                .GET()
                .build();
        return send(req, 200);
    }

    private Map<String, Object> post(String path, Map<String, Object> body, int... okStatuses) {
        byte[] raw;
        try {
            raw = mapper.writeValueAsBytes(body);
        } catch (IOException e) {
            throw new WorkerException.Rejected("无法序列化请求体: " + e.getMessage(), null);
        }
        HttpRequest req = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + path))
                .timeout(requestTimeout)
                .header("Content-Type", "application/json; charset=utf-8")
                .POST(HttpRequest.BodyPublishers.ofByteArray(raw))
                .build();
        return send(req, okStatuses);
    }

    private Map<String, Object> send(HttpRequest req, int... okStatuses) {
        HttpResponse<String> resp;
        try {
            resp = http.send(req, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        } catch (IOException e) {
            // 连接被拒 / 连接重置：worker 没起来或被强杀。
            // 这是最需要如实上报的一种失败 —— 绝不能退化成"返回空结果"。
            throw new WorkerException.Unavailable("无法连接 Python worker: " + e.getMessage(), null);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new WorkerException.Unavailable("请求 worker 被中断", null);
        }

        int status = resp.statusCode();
        for (int ok : okStatuses) {
            if (status == ok) {
                return parse(resp.body(), status);
            }
        }
        Map<String, Object> body = parseOrEmpty(resp.body());
        switch (status) {
            case 400 -> throw new WorkerException.Rejected(
                    str(body, "message", "worker 拒绝了请求"), body);
            case 404 -> throw new WorkerException.TaskNotFound(
                    str(body, "message", "worker 上不存在该任务"), body);
            case 409 -> {
                if("source_integrity_error".equals(body.get("error")))throw new WorkerException.SourceIntegrity(str(body,"message","Source integrity failed"),body);
                throw new WorkerException.Unexpected(
                    409, str(body, "message", "任务已处于终态"), body);
            }
            case 429 -> throw new WorkerException.QueueFull(
                    str(body, "message", "任务队列已满"), body);
            case 503 -> throw new WorkerException.Unavailable(
                    str(body, "message", "worker 不可用"), body);
            default -> throw new WorkerException.Unexpected(
                    status, "worker 返回未预期的状态 " + status + ": " + resp.body(), body);
        }
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> parse(String raw, int status) {
        if (raw == null || raw.isBlank()) {
            throw new WorkerException.Unavailable("Empty worker response", null);
        }
        try {
            return mapper.readValue(raw, Map.class);
        } catch (IOException e) {
            // 解析失败不该把 200 变成崩溃，但也不能假装什么都没发生
            log.warn("worker 响应无法解析: {}", e.getMessage());
            throw new WorkerException.Unavailable("Malformed worker JSON", null);
        }
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> parseOrEmpty(String raw) {
        if (raw == null || raw.isBlank()) {
            return Map.of();
        }
        try {
            return mapper.readValue(raw, Map.class);
        } catch (IOException e) {
            return Map.of("raw", raw);
        }
    }

    private static String str(Map<String, Object> body, String key, String fallback) {
        Object v = body.get(key);
        return v == null ? fallback : String.valueOf(v);
    }

    /**
     * 释放资源。
     *
     * 为什么没有 close()：JDK 的 HttpClient 没有 close 方法，
     * 它内部的 selector 线程会在没有引用后被 GC 回收。
     * 这里真正要关的是我们自己起的调度线程（在 TaskService 里），
     * 那个如果不关，JVM 进程就退不干净。
     */
    @PreDestroy
    public void shutdown() {
        http.shutdownNow();
    }
}
