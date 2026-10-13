package com.chenxinyu.docmind;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.test.context.DynamicPropertySource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Java 服务边界测试。
 *
 * 这些测试回答的是"服务在边界上怎么做"，不是"worker 内部怎么跑"：
 * 状态码有没有说实话、SSE 会不会永久悬挂、取消的两种语义有没有分清楚。
 * worker 内部行为由 Python 侧测试覆盖，真实链路由端到端脚本覆盖。
 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
        properties = {
                // 这里明确声明对面是桩：默认 real 下接桩后端必须被拒绝（另有测试验证）
                "docmind.expected-backend=stub",
                "docmind.sse.poll-interval-ms=120",
        })
class ServiceBoundaryTests {

    static FakeWorker worker;

    @Autowired
    TestRestTemplate rest;
    @Autowired
    ObjectMapper mapper;
    @LocalServerPort
    int port;

    @BeforeAll
    static void startFakeWorker() throws Exception {
        worker = new FakeWorker();
    }

    @AfterAll
    static void stopFakeWorker() {
        worker.stop();
    }

    @DynamicPropertySource
    static void props(org.springframework.test.context.DynamicPropertyRegistry registry) {
        registry.add("docmind.worker.base-url", () -> worker.baseUrl());
    }

    private String url(String path) {
        return "http://127.0.0.1:" + port + path;
    }

    private ResponseEntity<String> postJson(String path, String body) {
        HttpHeaders h = new HttpHeaders();
        h.setContentType(MediaType.APPLICATION_JSON);
        return rest.postForEntity(url(path), new HttpEntity<>(body, h), String.class);
    }

    // ------------------------------------------------------------------

    @Test
    void submit_returns_202_with_task_id_and_urls() throws Exception {
        ResponseEntity<String> r = postJson("/api/tasks", "{\"question\":\"sinh 正则化收敛速度？\"}");
        assertEquals(HttpStatus.ACCEPTED, r.getStatusCode());
        JsonNode n = mapper.readTree(r.getBody());
        assertNotNull(n.get("task_id").asText());
        assertEquals("queued", n.get("status").asText());
        assertTrue(n.get("status_url").asText().contains(n.get("task_id").asText()));
        assertTrue(n.get("events_url").asText().endsWith("/events"));
    }

    @Test
    void empty_and_blank_question_rejected_with_400() {
        assertEquals(HttpStatus.BAD_REQUEST, postJson("/api/tasks", "{\"question\":\"\"}").getStatusCode());
        assertEquals(HttpStatus.BAD_REQUEST, postJson("/api/tasks", "{\"question\":\"   \"}").getStatusCode());
        assertEquals(HttpStatus.BAD_REQUEST, postJson("/api/tasks", "{}").getStatusCode());
        assertEquals(HttpStatus.BAD_REQUEST, postJson("/api/tasks", "{\"question\":123}").getStatusCode());
    }

    @Test
    void too_long_question_rejected_with_400() {
        String q = "x".repeat(2001);
        ResponseEntity<String> r = postJson("/api/tasks", "{\"question\":\"" + q + "\"}");
        assertEquals(HttpStatus.BAD_REQUEST, r.getStatusCode());
        assertTrue(r.getBody().contains("过长"));
    }

    @Test
    void token_budget_out_of_range_rejected() {
        assertEquals(HttpStatus.BAD_REQUEST,
                postJson("/api/tasks", "{\"question\":\"a\",\"token_budget\":10}").getStatusCode());
        assertEquals(HttpStatus.BAD_REQUEST,
                postJson("/api/tasks", "{\"question\":\"a\",\"token_budget\":99999}").getStatusCode());
        // 布尔值不能被当成 1 静默接受
        assertEquals(HttpStatus.BAD_REQUEST,
                postJson("/api/tasks", "{\"question\":\"a\",\"max_rounds\":true}").getStatusCode());
        // 合法边界应当通过
        assertEquals(HttpStatus.ACCEPTED,
                postJson("/api/tasks", "{\"question\":\"a\",\"token_budget\":256}").getStatusCode());
    }

    @Test
    void queue_full_maps_to_429_not_500() {
        worker.setMode(FakeWorker.Mode.QUEUE_FULL);
        try {
            ResponseEntity<String> r = postJson("/api/tasks", "{\"question\":\"a\"}");
            assertEquals(HttpStatus.TOO_MANY_REQUESTS, r.getStatusCode());
            assertTrue(r.getBody().contains("queue_full"));
        } finally {
            worker.setMode(FakeWorker.Mode.NORMAL);
        }
    }

    @Test
    void worker_unavailable_maps_to_503() {
        worker.setMode(FakeWorker.Mode.UNAVAILABLE);
        try {
            ResponseEntity<String> r = postJson("/api/tasks", "{\"question\":\"a\"}");
            assertEquals(HttpStatus.SERVICE_UNAVAILABLE, r.getStatusCode());
            assertTrue(r.getBody().contains("worker_unavailable"));
            ResponseEntity<String> ready = rest.getForEntity(url("/api/ready"), String.class);
            assertEquals(HttpStatus.SERVICE_UNAVAILABLE, ready.getStatusCode());
        } finally {
            worker.setMode(FakeWorker.Mode.NORMAL);
        }
    }

    @Test
    void backend_mismatch_is_rejected_not_silently_accepted() {
        // 服务期望 stub，worker 却报 real —— 必须拒绝，避免任何"悄悄用假答案"的可能
        worker.setStub(false);
        try {
            ResponseEntity<String> r = postJson("/api/tasks", "{\"question\":\"a\"}");
            assertEquals(HttpStatus.SERVICE_UNAVAILABLE, r.getStatusCode());
            assertTrue(r.getBody().contains("后端不匹配"));
        } finally {
            worker.setStub(true);
        }
    }

    @Test
    void status_reaches_completed_and_carries_result() throws Exception {
        ResponseEntity<String> created = postJson("/api/tasks", "{\"question\":\"a\"}");
        String id = mapper.readTree(created.getBody()).get("task_id").asText();
        JsonNode view = null;
        for (int i = 0; i < 60; i++) {
            view = mapper.readTree(rest.getForEntity(url("/api/tasks/" + id), String.class).getBody());
            if ("completed".equals(view.get("status").asText())) {
                break;
            }
            Thread.sleep(100);
        }
        assertEquals("completed", view.get("status").asText());
        assertEquals("stub", view.get("backend").asText());
        assertTrue(view.get("stub").asBoolean(), "桩后端必须在结果里被标记出来");
        assertEquals("not_run", view.get("result").get("verification_status").asText());
    }

    @Test
    void failed_task_is_reported_as_failed_not_completed() throws Exception {
        worker.setMode(FakeWorker.Mode.FAIL_TASK);
        try {
            ResponseEntity<String> created = postJson("/api/tasks", "{\"question\":\"a\"}");
            String id = mapper.readTree(created.getBody()).get("task_id").asText();
            JsonNode view = null;
            for (int i = 0; i < 60; i++) {
                view = mapper.readTree(rest.getForEntity(url("/api/tasks/" + id), String.class).getBody());
                if ("failed".equals(view.get("status").asText())) {
                    break;
                }
                Thread.sleep(100);
            }
            assertEquals("failed", view.get("status").asText());
            assertNotNull(view.get("error"));
        } finally {
            worker.setMode(FakeWorker.Mode.NORMAL);
        }
    }

    @Test
    void cancel_distinguishes_queued_and_cooperative() throws Exception {
        // 第一次取消：任务还在 queued（假 worker 剧本里首次 GET 才转 running）
        ResponseEntity<String> created = postJson("/api/tasks", "{\"question\":\"a\"}");
        String id = mapper.readTree(created.getBody()).get("task_id").asText();
        ResponseEntity<String> c1 = rest.postForEntity(url("/api/tasks/" + id + "/cancel"), null, String.class);
        JsonNode n1 = mapper.readTree(c1.getBody());
        assertEquals("queued", n1.get("cancel_mode").asText());
        assertEquals("cancelled", n1.get("status").asText());

        // 第二次：让它跑起来再取消，必须是 cooperative 且状态仍是 running
        ResponseEntity<String> created2 = postJson("/api/tasks", "{\"question\":\"a\"}");
        String id2 = mapper.readTree(created2.getBody()).get("task_id").asText();
        rest.getForEntity(url("/api/tasks/" + id2), String.class);   // queued -> running
        ResponseEntity<String> c2 = rest.postForEntity(url("/api/tasks/" + id2 + "/cancel"), null, String.class);
        JsonNode n2 = mapper.readTree(c2.getBody());
        assertEquals("cooperative", n2.get("cancel_mode").asText());
        assertNotEquals("cancelled", n2.get("status").asText(),
                "运行中的任务不能声称已取消 —— 只能承诺在阶段边界停止");
    }

    @Test
    void sse_stream_reaches_terminal_event() throws Exception {
        ResponseEntity<String> created = postJson("/api/tasks", "{\"question\":\"a\"}");
        String id = mapper.readTree(created.getBody()).get("task_id").asText();

        HttpClient client = HttpClient.newHttpClient();
        HttpRequest req = HttpRequest.newBuilder(URI.create(url("/api/tasks/" + id + "/events")))
                .timeout(Duration.ofSeconds(20)).GET().build();
        CompletableFuture<HttpResponse<String>> future =
                client.sendAsync(req, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));

        // 轮询直到流结束
        HttpResponse<String> resp = future.get(25, TimeUnit.SECONDS);
        String body = resp.body();
        assertTrue(body.contains("event:"), "SSE 必须带事件类型");
        assertTrue(body.contains("terminal"), "流必须以终态事件结束，而不是永久悬挂");
        List<String> ids = body.lines().filter(l -> l.startsWith("id:"))
                .map(l -> l.substring(3).trim()).collect(Collectors.toList());
        assertTrue(ids.size() >= 2, "每条事件都必须有 id，客户端才能断线重连");
    }

    @Test
    void sse_client_disconnect_does_not_leak_subscriptions() throws Exception {
        ResponseEntity<String> created = postJson("/api/tasks", "{\"question\":\"a\"}");
        String id = mapper.readTree(created.getBody()).get("task_id").asText();

        HttpClient client = HttpClient.newBuilder().build();
        HttpRequest req = HttpRequest.newBuilder(URI.create(url("/api/tasks/" + id + "/events"))).GET().build();
        var stream = client.send(req, HttpResponse.BodyHandlers.ofLines());
        // 只读一行就断开，模拟浏览器关页面
        stream.body().findFirst();
        stream.body().close();

        int active = -1;
        for (int i = 0; i < 40; i++) {
            JsonNode h = mapper.readTree(rest.getForEntity(url("/api/health"), String.class).getBody());
            active = h.get("tasks").get("active_sse_subscriptions").asInt();
            if (active == 0) {
                break;
            }
            Thread.sleep(100);
        }
        assertEquals(0, active, "客户端断开后订阅必须被回收，否则线程/内存都会泄漏");
    }

    @Test
    void fractional_and_wrapped_integer_budgets_are_rejected() {
        for(String value:List.of("300.5", "4294967596")) {
            var r=postJson("/api/tasks", "{\"question\":\"a\",\"token_budget\":"+value+"}");
            assertEquals(HttpStatus.BAD_REQUEST,r.getStatusCode());
        }
    }
    @Test
    void two_observers_of_one_task_both_receive_terminal() throws Exception {
        var created=postJson("/api/tasks","{\"question\":\"a\"}");
        String id=mapper.readTree(created.getBody()).get("task_id").asText();
        try(var client=HttpClient.newHttpClient()) {
            var request=HttpRequest.newBuilder(URI.create(url("/api/tasks/"+id+"/events")))
                .timeout(Duration.ofSeconds(15)).GET().build();
            var first=client.sendAsync(request,HttpResponse.BodyHandlers.ofString());
            var second=client.sendAsync(request,HttpResponse.BodyHandlers.ofString());
            assertTrue(first.get(20,TimeUnit.SECONDS).body().contains("event:terminal"));
            assertTrue(second.get(20,TimeUnit.SECONDS).body().contains("event:terminal"));
        }
    }
    @Test
    void history_api_preserves_question_and_status() throws Exception {
        var response=rest.getForEntity(url("/api/tasks"),String.class);
        assertEquals(HttpStatus.OK,response.getStatusCode());
        var first=mapper.readTree(response.getBody()).get("tasks").get(0);
        assertEquals("prior question",first.get("question").asText());
        assertEquals("completed",first.get("status").asText());
    }
    @Test
    void upload_forwards_pdf_and_unicode_name() throws Exception {
        var headers=new HttpHeaders();headers.setContentType(MediaType.APPLICATION_PDF);
        headers.set("X-File-Name",java.net.URLEncoder.encode("测试论文.pdf",java.nio.charset.StandardCharsets.UTF_8));
        byte[] bytes="%PDF-1.4 example".getBytes(java.nio.charset.StandardCharsets.UTF_8);
        var response=rest.postForEntity(url("/api/imports"),new HttpEntity<>(bytes,headers),String.class);
        assertEquals(HttpStatus.ACCEPTED,response.getStatusCode());
        var data=mapper.readTree(response.getBody());assertEquals("测试论文.pdf",data.get("file_name").asText());
        assertEquals(bytes.length,data.get("bytes").asInt());
        assertEquals("ready",mapper.readTree(rest.getForObject(url("/api/imports/upload_test"),String.class)).get("status").asText());
    }
    @Test
    void upload_size_limit_rejects_before_worker() {
        var headers=new HttpHeaders();headers.setContentType(MediaType.APPLICATION_PDF);headers.set("X-File-Name","large.pdf");
        var response=rest.postForEntity(url("/api/imports"),new HttpEntity<>(new byte[20*1024*1024+1],headers),String.class);
        assertEquals(HttpStatus.PAYLOAD_TOO_LARGE,response.getStatusCode());
    }
    @Test void multiple_scope_rejects_empty_and_duplicates(){
        for(var ids:java.util.List.of(java.util.List.of(),java.util.List.of("a","a"))){
            var response=rest.postForEntity(url("/api/tasks"),Map.of("question","q","document_ids",ids),String.class);
            assertEquals(HttpStatus.BAD_REQUEST,response.getStatusCode());
        }
    }
    @Test void page_image_mime_and_page_count_forwarded(){
        var response=rest.getForEntity(url("/api/documents/test/pages/1"),byte[].class);
        assertEquals(HttpStatus.OK,response.getStatusCode());assertEquals(MediaType.IMAGE_PNG,response.getHeaders().getContentType());
        assertEquals("2",response.getHeaders().getFirst("X-PDF-Pages"));assertEquals((byte)0x89,response.getBody()[0]);
        assertEquals(HttpStatus.BAD_REQUEST,rest.getForEntity(url("/api/documents/test/pages/0"),String.class).getStatusCode());
    }
    @Test void changed_source_rejected_without_image(){
        for(String path:java.util.List.of("/api/documents/changed/pages/1","/api/documents/changed/pdf")){
            var response=rest.getForEntity(url(path),String.class);
            assertEquals(HttpStatus.CONFLICT,response.getStatusCode());assertTrue(response.getBody().contains("source_integrity_error"));
        }
    }
}
