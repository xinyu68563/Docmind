package com.chenxinyu.docmind.web;

import java.util.LinkedHashMap;
import java.util.Map;

import com.chenxinyu.docmind.config.DocMindProperties;
import com.chenxinyu.docmind.task.TaskService;
import com.chenxinyu.docmind.worker.WorkerClient;
import com.chenxinyu.docmind.worker.WorkerException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

/**
 * 健康 / 就绪。
 *
 * 【为什么 health 和 ready 要分开】
 *   health = 进程活着（ JVM 起来了、端口在听）
 *   ready  = 能真正干活（Python worker 可达且已经加载好模型）
 *   混成一个的话，编排系统会在模型还没加载完时就开始往里打流量，
 *   然后所有请求一起超时。
 *
 * 【backend / stub 必须显式暴露】
 *   桩后端只能用于打通链路。默认期望 real，
 *   一旦发现对面是 stub 就明确标出来，绝不让假答案混进生产路径。
 */
@RestController
@RequestMapping("/api")
public class HealthController {

    private static final Logger log = LoggerFactory.getLogger(HealthController.class);

    private final WorkerClient worker;
    private final TaskService tasks;
    private final DocMindProperties props;

    public HealthController(WorkerClient worker, TaskService tasks, DocMindProperties props) {
        this.worker = worker;
        this.tasks = tasks;
        this.props = props;
    }

    @GetMapping("/health")
    public ResponseEntity<Map<String, Object>> health() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("service", "docmind-service");
        out.put("status", "up");

        Map<String, Object> w;
        try {
            w = worker.health();
        } catch (WorkerException.Unavailable e) {
            w = Map.of("reachable", false, "reason", String.valueOf(e.getMessage()));
        } catch (RuntimeException e) {
            w = Map.of("reachable", false, "reason", e.toString());
        }
        out.put("worker", w);

        Object backend = w.get("backend");
        boolean stub = Boolean.TRUE.equals(w.get("stub")) || "stub".equals(backend);
        // 期望 real 而对面上来就是 stub（或反过来）都算不匹配。
        // 这是"禁止生产悄悄 fallback 假答案"的机器可查证据。
        boolean mismatch = !props.expectedBackend().equals(backend);
        out.put("expected_backend", props.expectedBackend());
        out.put("backend", backend);
        out.put("stub", stub);
        out.put("backend_mismatch", mismatch);
        if (mismatch) {
            log.warn("后端不匹配：服务期望 {}，worker 实际为 {}。桩后端绝不能用于生产路径。",
                    props.expectedBackend(), backend);
        }

        out.put("tasks", Map.of(
                "known", tasks.knownTasks(),
                "active_sse_subscriptions", tasks.activeSubscriptions()));
        out.put("limits", Map.of(
                "max_question_chars", props.tasks().maxQuestionChars(),
                "token_budget", Map.of("min", props.tasks().minTokenBudget(),
                        "max", props.tasks().maxTokenBudget())));
        // 明确写出"任务状态不持久"，避免调用方误以为服务重启后还能查到
        out.put("persistence", w.getOrDefault("persistence", Map.of("enabled",false)));

        HttpStatus code = Boolean.FALSE.equals(w.get("reachable")) || mismatch || !Boolean.TRUE.equals(w.get("ready")) ? HttpStatus.SERVICE_UNAVAILABLE : HttpStatus.OK;
        return ResponseEntity.status(code).body(out);
    }

    @GetMapping("/ready")
    public ResponseEntity<Map<String, Object>> ready() {
        Map<String, Object> w;
        try {
            w = worker.health();
        } catch (RuntimeException e) {
            return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                    .body(Map.of("ready", false, "reason", String.valueOf(e.getMessage())));
        }
        boolean ready = Boolean.TRUE.equals(w.get("ready")) && Boolean.TRUE.equals(w.get("executor_alive"))
                && props.expectedBackend().equals(w.get("backend"));
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("ready", ready);
        body.put("worker", w);
        return ResponseEntity.status(ready ? HttpStatus.OK : HttpStatus.SERVICE_UNAVAILABLE).body(body);
    }
}
