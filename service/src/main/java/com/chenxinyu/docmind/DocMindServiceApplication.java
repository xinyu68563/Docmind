package com.chenxinyu.docmind;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

/**
 * Phase 4 最小服务化的入口。
 *
 * 【这个服务到底负责什么】
 *
 *   Java 侧**不做**任何检索、推理、模型相关的事。它的职责只有四条：
 *     1. 校验用户输入（长度、预算范围），把不合法的请求挡在门口
 *     2. 把任务交给常驻 Python worker，拿回 task_id
 *     3. 把 worker 的真实状态翻译成 HTTP 状态 / SSE 事件
 *     4. 在 worker 不可用或队列满时给出明确的状态码，绝不伪造成功
 *
 * 【为什么 Java 不直接调 Python 的库】
 *
 *   模型是 PyTorch + CUDA 的，Java 生态里没有等价的本地推理路径。
 *   硬要用 JNI / Py4J 把 7B 模型塞进 JVM，只会得到两个都跑不好的运行时。
 *   "两个进程 + 一个窄接口"是这里唯一合理的边界：
 *   窄接口意味着两边可以独立测试、独立重启、独立演进。
 */
@SpringBootApplication
@ConfigurationPropertiesScan
public class DocMindServiceApplication {

    public static void main(String[] args) {
        SpringApplication.run(DocMindServiceApplication.class, args);
    }
}
