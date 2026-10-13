package com.chenxinyu.docmind.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * 服务侧可调参数。
 *
 * 用 record 而不是可变 POJO：这些值启动后就不该变，
 * 不可变能避免"运行到一半被改了"这类最难查的 bug。
 */
@ConfigurationProperties(prefix = "docmind")
public record DocMindProperties(
        Worker worker,
        Tasks tasks,
        Sse sse,
        String expectedBackend) {

    public record Worker(
            String baseUrl,
            int connectTimeoutMs,
            int requestTimeoutMs) {
    }

    public record Tasks(
            int maxQuestionChars,
            int minTokenBudget,
            int maxTokenBudget,
            int maxRoundsMax) {
    }

    public record Sse(
            long pollIntervalMs,
            long timeoutMs,
            int replayBuffer) {
    }
}
