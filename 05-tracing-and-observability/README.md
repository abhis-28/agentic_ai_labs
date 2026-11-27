
### 🔍 Tracing & Observability: debugging a support bot with 5 approaches
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abhis-28/agentic_ai_labs/blob/main/05-tracing-and-observability/Tracing_Observability_Lab.ipynb)

Observability = being able to answer "why did the LLM say that?" by reading a trace instead of guessing. This lab follows one real scenario: a support bot for a fictional cloud-storage company, CloudNest, tells a customer the free plan gives 5 GB when the real answer is 15 GB. The product docs contain a discontinued plan once marketed as "Free", worded closely enough that a real dense retriever (`all-MiniLM-L6-v2`) can pull the wrong chunk. The same bug is then traced with five approaches, using real Groq calls (`openai/gpt-oss-20b`).

**What it showed: same bug, five ways to trace it:**

| Part | Approach | Used for |
|---|---|---|
| 1 | Naive pass/fail test | 5 answers (one per doc chunk, plus no context) |
| 2 | Own tracer (spans, trace tree) | Finding which chunk was retrieved |
| 3 | LangSmith | Two-tool LangGraph agent, traced with env variables only |
| 4 | Langfuse | Raw Groq SDK chatbot, filtered per customer and session |
| 5 | OpenTelemetry + OpenInference | Raw span format on the console |
| 6 | Arize Phoenix | 5 questions, before vs. after fixing the retriever |

- **A shape-only test can't catch a wrong answer.** [FILL: the PASS/FAIL line for `legacy_starter` (5 GB) and `current_free` (15 GB) from Part 1]
- **A hand-built trace answered the manager's question.** [FILL: similarity scores for all 4 chunks and which chunk won, from the Part 2 tree]
- **Raw spans are the format under every tool.** [FILL: fields seen in the Part 5 console output, e.g. model name, token counts]
- **The fix was measured at scale.** [FILL: how many of the 5 questions picked `legacy_starter` before vs. after tagging it as discontinued, and any question still wrong]

**Caveats:** [FILL: anything that failed or looked off in your run]. The LangSmith and Langfuse results live in their dashboards, not in the notebook, so the notebook itself doesn't show them. The batch is 5 questions, so the before/after counts illustrate the idea and are not statistics.

**Setup:** Groq key as `agentic_ai` in Colab secrets, plus free accounts for LangSmith (`langsmith_api_key`) and Langfuse (`langfuse_public_key`, `langfuse_secret_key`). Parts 2, 5 and 6 need no account. No GPU needed.
