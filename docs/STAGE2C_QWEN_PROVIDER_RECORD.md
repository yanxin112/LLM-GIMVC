# Stage 2C Qwen Provider Record

## Goal

Use Alibaba Cloud Model Studio / DashScope Qwen as the real LLM and embedding provider for semantic view recovery.

## Providers

LLM:
- qwen

Embedding:
- qwen

## Recommended Models

LLM:
- qwen-plus
- qwen-turbo

Embedding:
- text-embedding-v4

## Environment

Set API key:

export DASHSCOPE_API_KEY="..."

Default base URL:

https://dashscope.aliyuncs.com/compatible-mode/v1

Alternative international base URL:

https://dashscope-intl.aliyuncs.com/compatible-mode/v1

## Embedding Dimension

text-embedding-v4 supports custom dimensions including 512.

The implementation sets:

dimensions=512

to match LLM-GIMVC latent_dim.

## Safety

- DASHSCOPE_API_KEY is read from environment.
- API key is never logged.
- --max-llm-queries controls cost.
- partial outputs are not safe_for_fusion.

## Smoke Command

python run_llm_semantic_path.py \
  --dataset BDGP \
  --missing-rate 0.5 \
  --seed 0 \
  --provider qwen \
  --embedding-provider qwen \
  --llm-model qwen-plus \
  --embedding-model text-embedding-v4 \
  --embedding-dim 512 \
  --max-llm-queries 5 \
  --device cpu
