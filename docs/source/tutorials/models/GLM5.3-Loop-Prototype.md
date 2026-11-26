# GLM-5.3 权重的 Loop 原型

用完整版 GLM-5.3 的 `GlmMoeDsaForCausalLM` checkpoint 运行 GLM-5.4 的文本 Loop 拓扑。`Glm53LoopForCausalLM` 复用 78 层的权重和现有 Ascend loader。L1–26 执行一遍，L27–54 共用权重执行两遍，L55–78 执行一遍。

第二轮按 query 复用第一轮的七组 top-k，读取同一物理层的第一轮全局 MLA KV，再加入独立缓存中最近 128 个位置的本轮 KV。两路一起计算 softmax，位置相同的两轮 KV 各占一个条目。当前实现为两轮分配独立的分页缓存。第二轮读取受滑窗限制，缓存仍按完整序列分配。

模型接受 `glm53_loop_gates` 两个标量，分别应用于两轮的写回 `x + gate * (h - x)`，默认 `[1.0, 1.0]`。这两个值模拟数据流。GLM-5.3 checkpoint 没有 GLM-5.4 训练得到的逐 token 门控参数。

## 启动

使用本分支和 `.github/vllm-main-verified.commit` 指定的 vLLM 版本。以下为单机 Atlas A3、16 个逻辑设备的参考命令。`MODEL_PATH` 指向 [GLM-5.3 W8A8C8 checkpoint](https://www.modelscope.cn/models/Eco-Tech/GLM-5.3-w8a8c8)，模型权重使用 Ascend 量化加载，注意力 KV 用 BF16。

原型要求 eager、单实例 target 推理、PP/DP/PCP/DCP 均为 1，上下文最多 4096 token。启动时关闭 prefix caching。以下命令保留 TP16 和 EP，单请求先验证 1024 token 内的 prefill 和 decode。

```bash
cd /workspace
export MODEL_PATH=/path/to/GLM-5.3-w8a8c8
export HCCL_BUFFSIZE=200
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V2_MODEL_RUNNER=1

vllm serve "$MODEL_PATH" \
  --served-model-name glm53-loop \
  --hf-overrides '{"architectures":["Glm53LoopForCausalLM"],"glm53_loop_gates":[1.0,1.0]}' \
  --trust-remote-code --quantization ascend --dtype bfloat16 --kv-cache-dtype auto \
  --tensor-parallel-size 16 --data-parallel-size 1 --enable-expert-parallel \
  --enforce-eager --no-enable-prefix-caching \
  --max-model-len 1024 --max-num-batched-tokens 128 --max-num-seqs 1 \
  --port 8000
```

## 验证

```bash
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"glm53-loop","prompt":"The capital of France is","temperature":0,"max_tokens":8}'
```

首个请求应返回 HTTP 200 和非空生成结果。短 prompt 验证 prefill 与连续 decode。用超过 128 token 的 prompt 再验证一次分块 prefill 和滑窗边界。准确率和吞吐需要实卡测量，原型的输出不能用来推断 GLM-5.4 的效果。
