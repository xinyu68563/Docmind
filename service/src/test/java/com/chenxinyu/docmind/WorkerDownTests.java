package com.chenxinyu.docmind;

import java.util.Map;

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
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * worker 进程真的不在了（连接被拒）时的行为。
 *
 * 为什么单独开一个测试类：
 *   这个场景要的是"端口上没人听"，必须先把假 worker 停掉。
 *   停掉之后同一个端口没法重用，所以只能另起一个 Spring 上下文，
 *   让它指向一个从一开始就死掉的地址。
 *
 * 这一条对应的真实事故：Python worker 崩了 / 被强杀 / 还没启动，
 * Java 服务必须明确报 503，绝不能返回 200 + 空结果。
 */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT,
        properties = "docmind.expected-backend=stub")
class WorkerDownTests {

    static String deadUrl;

    @Autowired
    TestRestTemplate rest;
    @LocalServerPort
    int port;

    @BeforeAll
    static void startThenKillWorker() throws Exception {
        FakeWorker w = new FakeWorker();
        deadUrl = w.baseUrl();
        w.stop();                     // 端口上从此没人听
    }

    @DynamicPropertySource
    static void props(org.springframework.test.context.DynamicPropertyRegistry registry) {
        registry.add("docmind.worker.base-url", () -> deadUrl);
    }

    @Test
    void submit_returns_503_when_worker_process_is_gone() {
        HttpHeaders h = new HttpHeaders();
        h.setContentType(MediaType.APPLICATION_JSON);
        ResponseEntity<String> r = rest.postForEntity(
                "http://127.0.0.1:" + port + "/api/tasks",
                new HttpEntity<>("{\"question\":\"a\"}", h), String.class);
        assertEquals(HttpStatus.SERVICE_UNAVAILABLE, r.getStatusCode());
        assertTrue(r.getBody().contains("worker_unavailable"));
    }

    @Test
    void health_marks_worker_unreachable() {
        ResponseEntity<String> r = rest.getForEntity(
                "http://127.0.0.1:" + port + "/api/health", String.class);
        assertEquals(HttpStatus.SERVICE_UNAVAILABLE, r.getStatusCode());
        assertTrue(r.getBody().contains("reachable"));
    }

    @Test
    void ready_is_503_when_worker_is_gone() {
        ResponseEntity<String> r = rest.getForEntity(
                "http://127.0.0.1:" + port + "/api/ready", String.class);
        assertEquals(HttpStatus.SERVICE_UNAVAILABLE, r.getStatusCode());
    }
}
