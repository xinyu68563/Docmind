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
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

/**
 * 用户 API。
 *
 * 设计原则：每个状态码都要能在文档里说出"它意味着调用方下一步该做什么"。
 *   202 已受理（异步执行，去拿 task_id）
 *   400 你说的话我不接受（改请求）
 *   429 我忙不过来（稍后重试）
 *   503 我依赖的 worker 不可用（需要人介入）
 * 绝不用 200 + body 里的 error 字段来糊过去 —— 那等于让客户端
 * 必须解析 body 才知道成败，是 REST 里最典型的反模式。
 */
@RestController
@RequestMapping("/api")
public class TaskController {

    private static final Logger log = LoggerFactory.getLogger(TaskController.class);

    private final TaskService tasks;
    private final DocMindProperties props;
    private final WorkerClient worker;

    public TaskController(TaskService tasks, DocMindProperties props, WorkerClient worker) {
        this.tasks = tasks;
        this.props = props;
        this.worker = worker;
    }

    /**
     * 提交问题任务。
     *
     * 为什么是 202 而不是 200：这个请求只是"受理"，答案还没生成。
     * 返回 200 会让客户端以为已经拿到结果。
     */
    @PostMapping(path = "/tasks", consumes = MediaType.APPLICATION_JSON_VALUE)
    public ResponseEntity<Map<String, Object>> submit(@RequestBody(required = false) Map<String, Object> body) {
        Map<String, Object> validated = validate(body);
        assertBackendAllowed();
        Map<String, Object> created = tasks.submit(validated);
        String taskId = String.valueOf(created.get("task_id"));

        Map<String, Object> out = new LinkedHashMap<>(created);
        out.put("status_url", "/api/tasks/" + taskId);
        out.put("events_url", "/api/tasks/" + taskId + "/events");
        out.put("cancel_url", "/api/tasks/" + taskId + "/cancel");
        return ResponseEntity.status(HttpStatus.ACCEPTED).body(out);
    }

    @GetMapping("/tasks")
    public Map<String,Object> list(){return worker.tasks();}

    @GetMapping("/tasks/{taskId}")
    public Map<String, Object> status(@PathVariable String taskId) {
        return tasks.status(taskId);
    }

    /**
     * 取消任务。
     *
     * 返回体里的 cancel_mode 说明了这次取消的**真实**能力边界：
     *   queued      —— 任务还没开始，一次都不会运行
     *   cooperative —— 已在运行，将在下一个阶段边界停止；
     *                  正在进行的模型生成无法立即中断，GPU 仍在干活
     * 这两者必须区分，否则"取消成功"会变成一句空话。
     */
    @PostMapping("/tasks/{taskId}/cancel")
    public Map<String, Object> cancel(@PathVariable String taskId) {
        return tasks.cancel(taskId);
    }

    /**
     * SSE 事件流。
     *
     * produces = text/event-stream 是必须的，否则客户端不会按 SSE 解析。
     */
    @GetMapping(path = "/tasks/{taskId}/events", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter events(@PathVariable String taskId,
                             @RequestHeader(value = "Last-Event-ID", required = false) String lastEventId) {
        return tasks.subscribe(taskId, lastEventId);
    }

    /**
     * 后端一致性闸门：期望 real 时绝不接受 stub 后端提交的答案。
     *
     * 为什么是"拒绝"而不是"打个日志继续"：
     *   桩后端返回的是假答案。一旦允许它在默认配置下流通，
     *   调用方无法从响应上分辨真假（它是有意做得像真的）。
     *   安全默认必须是"不一致就不服务"，要跑桩就得显式声明 expected-backend=stub。
     */
    private void assertBackendAllowed() {
        Map<String, Object> h;
        try {
            h = worker.health();
        } catch (WorkerException.Unavailable e) {
            throw e;
        } catch (RuntimeException e) {
            throw new WorkerException.Unavailable("健康检查失败: " + e.getMessage(), null);
        }
        if (!props.expectedBackend().equals(h.get("backend")) || !Boolean.TRUE.equals(h.get("ready"))) {
            throw new WorkerException.Unavailable("后端不匹配或 worker 不可用", h);
        }
        boolean stub = Boolean.TRUE.equals(h.get("stub")) || "stub".equals(h.get("backend"));
        if (stub != "stub".equalsIgnoreCase(props.expectedBackend())) {
            throw new WorkerException.Unavailable(
                    "后端不匹配：服务期望 " + props.expectedBackend()
                            + "，worker 实际为 " + h.get("backend")
                            + "；拒绝提交，以免把桩结果当成真实答案", h);
        }
    }

    // ------------------------------------------------------------------
    // 输入校验
    // ------------------------------------------------------------------

    /**
     * 为什么手动校验而不用 Bean Validation：
     *   spring-boot-starter-validation 会额外拉一整套 hibernate-validator，
     *   而这里只有四个字段。更重要的是——校验规则必须和 Python worker 侧
     *   逐字对齐（同一个上限），写在同一个方法里反而更容易保证一致。
     */
    private Map<String, Object> validate(Map<String, Object> body) {
        if (body == null) {
            throw new WorkerException.Rejected("请求体不能为空", null);
        }
        Object q = body.get("question");
        if (!(q instanceof String s) || s.isBlank()) {
            throw new WorkerException.Rejected("question 必须是非空字符串", Map.of("field", "question"));
        }
        String question = s.strip();
        if (question.length() > props.tasks().maxQuestionChars()) {
            throw new WorkerException.Rejected(
                    "question 过长：" + question.length() + " > " + props.tasks().maxQuestionChars(),
                    Map.of("field", "question"));
        }

        Map<String, Object> out = new LinkedHashMap<>();
        out.put("question", question);
        Object documentId=body.get("document_id");
        if(documentId!=null){
            if(!(documentId instanceof String id)||id.isBlank()||id.length()>100)
                throw new WorkerException.Rejected("document_id 必须是有效文档编号",Map.of("field","document_id"));
            out.put("document_id",documentId);
        }

        Object ids=body.get("document_ids");
        if(ids!=null){
            if(documentId!=null||!(ids instanceof java.util.List<?> list)||list.isEmpty()||list.size()>10||list.stream().anyMatch(x->!(x instanceof String str)||str.isBlank()||str.length()>100)||new java.util.HashSet<>(list).size()!=list.size())
                throw new WorkerException.Rejected("请选择1到10篇不同论文",Map.of("field","document_ids"));
            out.put("document_ids",ids);
        }
        Object budget = body.get("token_budget");
        if (budget != null) {
            out.put("token_budget", requireIntRange(budget, "token_budget",
                    props.tasks().minTokenBudget(), props.tasks().maxTokenBudget()));
        }
        Object rounds = body.get("max_rounds");
        if (rounds != null) {
            out.put("max_rounds", requireIntRange(rounds, "max_rounds", 0, props.tasks().maxRoundsMax()));
        }
        Object topK = body.get("top_k");
        if (topK != null) {
            out.put("top_k", requireIntRange(topK, "top_k", 1, 20));
        }
        return out;
    }

    /**
     * 整数范围校验。刻意拒绝布尔值：JSON 里 true 会被很多解析器当成 1，
     * 不挡住的话 {"max_rounds": true} 就变成了一次合法的 1 轮检索。
     */
    private int requireIntRange(Object v, String field, int lo, int hi) {
        if (v instanceof Boolean || !(v instanceof Number n)) {
            throw new WorkerException.Rejected(field + " 必须是整数", Map.of("field", field));
        }
        double value = n.doubleValue();
        if (!Double.isFinite(value) || value != Math.rint(value) || value < lo || value > hi) {
            throw new WorkerException.Rejected(field + " must be an integer in range", Map.of("field", field));
        }
        int i = n.intValue();
        if (i < lo || i > hi) {
            throw new WorkerException.Rejected(field + " 必须在 [" + lo + ", " + hi + "] 之间，收到 " + i,
                    Map.of("field", field));
        }
        return i;
    }
}
