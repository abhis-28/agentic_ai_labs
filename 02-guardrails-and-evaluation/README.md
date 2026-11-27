### 🛡️ Guardrails for LLM Agents: four real defenses tested on the same five attacks
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abhis-28/agentic_ai_labs/blob/main/Guardrails_Demo.ipynb)

One fictional customer-support chatbot (**ShopSmart**) and one shared set of 5 real test cases — a safe question, a prompt injection, a jailbreak attempt, a message containing PII, and a leading question trying to get the bot to confirm a false policy — run through **four different guardrail mechanisms**, so their real behavior can be compared side by side instead of just described:
1. **Prompt-based** — instructions only, in the system prompt.
2. **Llama Guard** — a dedicated safety-classifier model, called via Groq.
3. **Guardrails AI** — a validator framework (PII detection via Microsoft Presidio, jailbreak detection backed by Llama Guard).
4. **NeMo Guardrails** — a programmable input/output rail system using self-check flows.

Every call is a real LLM request — no mocked responses, no hardcoded verdicts.

**What it showed (5 tests, 4 guardrails):**
- **The baseline bot (no guardrails) already refused the injection and hacking attempts on its own**, and correctly stated the true 30-day policy rather than confirming the false 90-day claim it was led toward — a reminder that a capable base model isn't defenseless, even before any guardrail is added.
- **Prompt-based guardrails over-corrected:** they blocked the *safe* order-tracking question ("I don't have that information") while still catching the real injection and hacking attempts and correctly resisting the false-policy question — a real false positive, not a hypothetical one.
- **Llama Guard gave a clean separation:** ~0.0004–0.0006 probability of being unsafe for the three legitimate inputs, vs. 0.997–0.9996 for the injection and hacking attempts. On a genuinely harmful sample bot`guardrails-ai`, `presidio-analyzer`, `nemoguardrails`-

### 👔 Supervisor Pattern: static vs. dynamic, good vs. bad decomposition — on the same task
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abhis-28/agentic_ai_labs/blob/main/Supervisor_Pattern_Demo.ipynb)

One supervisor, two real specialist workers (a billing-policy worker and a live-web-search research worker), run as **four variants** — static or dynamic delegation timing, crossed with good or bad task decomposition — on the same two real questions. The only difference between "good" and "bad" is one instruction: good writes each worker a scoped sub-task containing only the part it needs; bad just forwards the user's entire question to every worker, unedited.

**What it showed (2 questions, 4 variants each):**
- **Q1 (needs only billing):** all four variants gave essentially the same correct refund answer. Decomposition quality genuinely didn't matter here — exactly as predicted, since there's only one worker to confuse.
- **Q2 (needs both billing and research), good decomposition:** clean, scoped sub-tasks (billing got only the refund question, research got only the outage question) produced a short, accurate combined answer in both static and dynamic mode.
- **Q2, bad decomposition — the static run exposed a real failure mode I hadn't predicted:** given the *entire* unscoped question, the research worker's live web search pulled up **OpenAI's actual ChatGPT subscription refund policy** (a 7-day discretionary refund window, real OpenAI terms) and wove it into the answer — contaminating a question about our fictional company's billing policy with a real, unrelated company's real policy. This is a concrete, unplanned example of why unscoped delegation is risky: it didn't just produce a vague answer, it let one worker's web search drag in genuinely wrong context.
- **The same "bad" instruction didn't always produce the same failure:** in the *dynamic* bad-decomposition run, the supervisor actually trimmed the research worker's sub-task to just the outage question anyway, despite being told to forward the question unchanged — and that run's answer stayed clean. Same degraded instruction, two different real outcomes, depending on static vs. dynamic timing.
- **The cost tradeoff held exactly as predicted:** static used **2 supervisor LLM calls** regardless of how many workers were needed (1 to plan everything upfront + 1 to synthesize). Dynamic used **3 calls** for the one-worker question and **4 calls** for the two-worker question — cost scales with delegation steps, not fixed like static.

**Caveats:** this is 2 questions, so the OpenAI-policy-contamination result is a real but single observation, not a reliable rate. LLMs don't always follow a "bad" instruction as badly as asked (seen directly in the dynamic+bad run above), so decomposition-quality experiments like this one need multiple runs before drawing firm conclusions from any single pass.

**Setup:** needs only a Groq API key (`groq_api` in Colab secrets) and a live internet connection in Colab for the real web-search tool. No GPU needed.
