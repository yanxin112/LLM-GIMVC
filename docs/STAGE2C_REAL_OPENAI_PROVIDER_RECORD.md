# Stage 2C Real OpenAI Provider Record

## Status

Stage 2C implements real OpenAI LLM and embedding providers.

## Providers

LLM:
- mock
- openai

Embedding:
- mock
- openai

## Models

Default LLM:
- configurable, e.g. gpt-4o-mini

Default embedding:
- text-embedding-3-small

## Embedding Dimension

For text-embedding-3 models, the implementation uses the dimensions parameter to align the output embedding dimension with the latent dimension.

Default target dimension:
- config["Module"]["d_model"] / config["Module"]["trans_dim"]

## Safety

- OPENAI_API_KEY is read from environment.
- API key is never written to logs.
- max_llm_queries controls cost.
- partial outputs are marked is_partial=true.
- partial outputs are not safe_for_fusion by default.

## Smoke Test

BDGP missing_50 seed_0 with max_llm_queries=10.

## Limitations

- Stage 2C-smoke is not a formal experiment.
- Partial LLM outputs cannot be used as paper evidence.
- Formal Block 1 requires full safe_for_fusion outputs.
- OpenAI API cost must be controlled.
