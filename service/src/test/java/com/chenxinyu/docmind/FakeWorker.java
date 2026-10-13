package com.chenxinyu.docmind;

import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.http.HttpClient;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.sun.net.httpserver.HttpServer;

/**
 * 测试用的假 worker。
 *
 * 【为什么用假 HTTP 服务而不是假装调用】
 *   Java 侧真正要测的是**边界行为**：状态码怎么映射、SSE 怎么结束、
 *   断开后订阅有没有被回收。这些和 Python 无关，用假服务能把故障分支
 *   （429 / 503 / 运行中取消）确定性地造出来 —— 这些分支在真实 worker 上
 *   要么很难触发，要么会真的去加载 7B 模型。
 *
 * 【它不是"镜像实现"】
 *   这里不复制任何业务逻辑，只按协议返回固定剧本，
 *   真实链路的端到端验证由 scripts/round13_e2e_check.py 走真进程完成。
 */
public class FakeWorker implements AutoCloseable {

    public enum Mode { NORMAL, QUEUE_FULL, UNAVAILABLE, FAIL_TASK }

    private final HttpServer server;
    private final ObjectMapper mapper = new ObjectMapper();
    private final Map<String, String> statuses = new ConcurrentHashMap<>();
    private final AtomicInteger seq = new AtomicInteger();
    private volatile Mode mode = Mode.NORMAL;
    private volatile boolean stub = true;

    public FakeWorker() throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext("/v1/health", ex -> json(ex, 200, health()));
        server.createContext("/v1/documents",ex->{
            if(ex.getRequestURI().getPath().contains("/changed/")){json(ex,409,Map.of("error","source_integrity_error","message","Source changed"));return;}
            if(ex.getRequestURI().getPath().endsWith("/pages/0")){json(ex,400,Map.of("message","bad page"));return;}
            byte[] bytes=java.util.Base64.getDecoder().decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jFoQAAAAASUVORK5CYII=");
            ex.getResponseHeaders().set("Content-Type","image/png");ex.getResponseHeaders().set("X-PDF-Pages","2");
            ex.sendResponseHeaders(200,bytes.length);try(var out=ex.getResponseBody()){out.write(bytes);}
        });
        server.createContext("/v1/imports",ex->{
            if("POST".equals(ex.getRequestMethod())){
                byte[] bytes=ex.getRequestBody().readAllBytes();
                String name=java.net.URLDecoder.decode(ex.getRequestHeaders().getFirst("X-File-Name"),StandardCharsets.UTF_8);
                json(ex,202,Map.of("doc_id","upload_test","status","queued","file_name",name,"bytes",bytes.length));
            }else json(ex,200,Map.of("doc_id","upload_test","status","ready"));
        });
        server.createContext("/v1/tasks", ex -> {
            String path = ex.getRequestURI().getPath();
            String method = ex.getRequestMethod();
            if ("GET".equals(method) && "/v1/tasks".equals(path)) {
                json(ex,200,Map.of("tasks",java.util.List.of(Map.of("task_id","history1","question","prior question","status","completed"))));
                return;
            }
            if ("POST".equals(method) && "/v1/tasks".equals(path)) {
                if (mode == Mode.QUEUE_FULL) {
                    json(ex, 429, Map.of("error", "queue_full", "queue", Map.of("size", 4, "max", 4)));
                    return;
                }
                if (mode == Mode.UNAVAILABLE) {
                    json(ex, 503, Map.of("error", "worker_unavailable", "message", "故障注入"));
                    return;
                }
                String id = "fake" + seq.incrementAndGet();
                statuses.put(id, "queued");
                json(ex, 202, Map.of("task_id", id, "status", "queued"));
                return;
            }
            String[] parts = path.split("/");
            if (parts.length >= 4 && "POST".equals(method) && "cancel".equals(parts[parts.length - 1])) {
                String id = parts[3];
                String st = statuses.getOrDefault(id, "queued");
                if ("queued".equals(st)) {
                    statuses.put(id, "cancelled");
                    json(ex, 200, Map.of("task_id", id, "status", "cancelled", "cancel_mode", "queued"));
                } else {
                    json(ex, 202, Map.of("task_id", id, "status", st, "cancel_mode", "cooperative"));
                }
                return;
            }
            if (parts.length >= 4 && "GET".equals(method)) {
                String id = parts[3];
                String st = statuses.getOrDefault(id, "queued");
                // 剧本：queued -> running -> 终态
                if ("queued".equals(st)) {
                    statuses.put(id, "running");
                    st = "running";
                } else if ("running".equals(st)) {
                    st = mode == Mode.FAIL_TASK ? "failed" : "completed";
                    statuses.put(id, st);
                }
                Map<String, Object> body = new java.util.LinkedHashMap<>();
                body.put("task_id", id);
                body.put("status", st);
                body.put("stage", st);
                body.put("events", java.util.List.of(
                        Map.of("seq", 1, "type", "created", "status", "queued", "stage", "queued"),
                        Map.of("seq", 2, "type", "status.running", "status", "running", "stage", "starting")));
                if ("completed".equals(st)) {
                    body.put("result", Map.of("answer", "stub answer", "backend", "stub",
                            "verification_status", "not_run"));
                }
                if ("failed".equals(st)) {
                    body.put("error", Map.of("error_type", "RuntimeError", "message", "boom"));
                }
                json(ex, 200, body);
                return;
            }
            json(ex, 404, Map.of("error", "not_found"));
        });
        server.setExecutor(java.util.concurrent.Executors.newFixedThreadPool(2));
        server.start();
    }

    public void setMode(Mode m) {
        this.mode = m;
    }

    public void setStub(boolean stub) {
        this.stub = stub;
    }

    public String baseUrl() {
        return "http://127.0.0.1:" + server.getAddress().getPort();
    }

    public void stop() {
        server.stop(0);
    }

    private Map<String, Object> health() {
        Map<String, Object> h = new java.util.LinkedHashMap<>();
        h.put("status", mode == Mode.UNAVAILABLE ? "unavailable" : "ready");
        h.put("ready", mode != Mode.UNAVAILABLE);
        h.put("backend", stub ? "stub" : "real");
        h.put("stub", stub);
        h.put("executor_alive", true);
        h.put("queue", Map.of("size", 0, "max", 4));
        return h;
    }

    private void json(com.sun.net.httpserver.HttpExchange ex, int code, Map<String, Object> body) {
        try {
            byte[] raw = mapper.writeValueAsBytes(body);
            ex.getResponseHeaders().add("Content-Type", "application/json; charset=utf-8");
            ex.sendResponseHeaders(code, raw.length);
            try (OutputStream os = ex.getResponseBody()) {
                os.write(raw);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        } finally {
            ex.close();
        }
    }

    @Override
    public void close() {
        stop();
    }

    static String read(String url) throws Exception {
        return HttpClient.newHttpClient().send(
                java.net.http.HttpRequest.newBuilder(java.net.URI.create(url)).GET().build(),
                java.net.http.HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8)).body();
    }
}
