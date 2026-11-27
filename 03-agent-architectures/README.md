
### 👔 Supervisor Pattern: static vs. dynamic, good vs. bad decomposition — on the same task
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abhis-28/agentic_ai_labs/blob/main/03-agent-architectures/Supervisor_Pattern_Demo.ipynb)

One supervisor, two real specialist workers (a billing-policy worker and a live-web-search research worker), run as **four variants** — static or dynamic delegation timing, crossed with good or bad task decomposition — on the same two real questions. The only difference between "good" and "bad" is one instruction: good writes each worker a scoped sub-task containing only the part it needs; bad just forwards the user's entire question to every worker, unedited.

**What it showed (2 questions, 4 variants each):**
- **Q1 (needs only billing):** all four variants gave essentially the same correct refund answer. Decomposition quality genuinely didn't matter here — exactly as predicted, since there's only one worker to confuse.
- **Q2 (needs both billing and research), good decomposition:** clean, scoped sub-tasks (billing got only the refund question, research got only the outage question) produced a short, accurate combined answer in both static and dynamic mode.
- **Q2, bad decomposition — the static run exposed a real failure mode I hadn't predicted:** given the *entire* unscoped question, the research worker's live web search pulled up **OpenAI's actual ChatGPT subscription refund policy** (a 7-day discretionary refund window, real OpenAI terms) and wove it into the answer — contaminating a question about our fictional company's billing policy with a real, unrelated company's real policy. This is a concrete, unplanned example of why unscoped delegation is risky: it didn't just produce a vague answer, it let one worker's web search drag in genuinely wrong context.
- **The same "bad" instruction didn't always produce the same failure:** in the *dynamic* bad-decomposition run, the supervisor actually trimmed the research worker's sub-task to just the outage question anyway, despite being told to forward the question unchanged — and that run's answer stayed clean. Same degraded instruction, two different real outcomes, depending on static vs. dynamic timing.
- **The cost tradeoff held exactly as predicted:** static used **2 supervisor LLM calls** regardless of how many workers were needed (1 to plan everything upfront + 1 to synthesize). Dynamic used **3 calls** for the one-worker question and **4 calls** for the two-worker question — cost scales with delegation steps, not fixed like static.

**Caveats:** this is 2 questions, so the OpenAI-policy-contamination result is a real but single observation, not a reliable rate. LLMs don't always follow a "bad" instruction as badly as asked (seen directly in the dynamic+bad run above), so decomposition-quality experiments like this one need multiple runs before drawing 

### 🐝 Swarm Pattern: peer-to-peer agent handoff, no central coordinator
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abhis-28/agentic_ai_labs/blob/main/03-agent-architectures/Swarm_Pattern_Demo.ipynb)

Three real peer agents — billing (policy knowledge), research (live web search), calculator (real refund arithmetic) — each deciding **on its own** whether to answer or hand off to another agent, by calling a real `handoff_to_X` tool. No supervisor ever routes the conversation; the full message history passes directly between agents on handoff. A loop guard stops the conversation after 4 handoffs if agents keep bouncing it back and forth.

**What it showed (4 real queries):**
- **Q1 (no handoff needed):** billing answered the refund-policy question directly, correctly, with 0 handoffs — exactly as designed.
- **Q2 (one handoff expected):** billing handed off to research for the live outage check, research ran a real web search, then handed control **back** to billing — but billing's final answer didn't actually report what research had found. Instead it said *"I'll check for any current OpenAI API outages right now"* (as if still about to check) and asked a redundant clarifying question about the user's plan type. This is a real, concrete example of swarm's **"no synthesis step"** weakness: whichever agent speaks last doesn't necessarily use what the other agent already learned.
- **Q3 (multi-hop expected — billing → calculator → research):** billing decided **no handoff was needed at all**, despite the question explicitly asking for an *exact* refund calculated from real numbers (`$20/month, 5 months used`) — exactly the scenario the calculator agent exists for. It never got invoked. On top of that, the final answer field came back **empty**. A genuine failure, not a designed outcome — and a useful one to have caught on camera rather than assumed away.
- **Q4 (loop-risk query):** billing handed off to research once; research then ran **three separate searches** on its own before answering directly, never handing back to billing despite the question being refund-adjacent. The loop guard was never triggered — not because it failed, but because the agents simply didn't bounce back and forth on this particular phrasing.

**Caveats:** Q3's empty answer and Q2's non-responsive final reply are real outputs from this run, not edge cases I constructed — they're a direct illustration of the tradeoff named in this notebook's own comparison section (no central synthesis means coherence isn't guaranteed). A different run, or a reworded Q3, might route through the calculator correctly; this notebook doesn't claim the failure is deterministic, only that it happened here and is worth knowing about before trusting a swarm architecture with multi-step numeric tasks.

**Setup:** needs only a Groq API key (`agentic_ai` in Colab secrets) and a live internet connection for the real web-search tool. No GPU needed.

### 🏢 Hierarchical Multi-Agent Pattern: CEO → Directors → workers, vs. one flat supervisor
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abhis-28/agentic_ai_labs/blob/main/03-agent-architectures/Hierarchical_Agent_Demo.ipynb)

The same real task (an NVIDIA AI-chip investment brief) run two ways: **hierarchically**, through a CEO delegating to 3 Directors who each manage their own real workers (web search + an LLM-based data-synthesis step), vs. **flat**, one supervisor managing all 7 workers directly. One reusable `run_team()` function (decompose → delegate → collect → synthesize) is called recursively for every level, so the only real difference between the two runs is the org chart.

**What it showed:**
- **A real decomposition bug, caught in the act:** inside the Analysis Director's own team, the LLM planned `data_synthesis` to run **before** `financial_analysis` — despite being explicitly instructed to order workers so dependents come after what they depend on. The result: `data_synthesis` ran with no financial data yet available and correctly reported *"No financial findings were provided... I'm unable to extract or summarize."* This is exactly the kind of coordination failure a hierarchy is supposed to make visible and localized — it broke inside one Director's team, not silently across the whole system.
- **Context size, measured directly:** the CEO's final synthesis call saw **15,831 characters** (3 Director summaries) vs. the flat supervisor's **27,034 characters** (7 raw/intermediate worker outputs) — hierarchical cut the top-level synthesis input to about 59% of the flat run's size, in the direction the notes predict.
- **The financial figures in both briefs are inconsistent with each other** (one run's financial worker reported a $115.8B quarterly revenue figure, the other reported ~$26B annual revenue) — a reminder that the real web-search-backed workers can retrieve or synthesize unreliable numbers, and this notebook's point is about context/coherence mechanics, not about trusting any specific figure it produces as real financial data.

**Caveats:** this notebook's two runs may have been captured under slightly different versions of the shared `run_team()` function (the flat run shows a context-trimming safeguard firing; the hierarchical run's saved output doesn't, despite exceeding the same threshold) — restart and re-run both before treating the character-count comparison as a strict apples-to-apples result. 7 workers is also smaller than the "~10 workers" scale the notes use to describe coherence breakdown, so a flat supervisor holding up reasonably well here doesn't contradict the notes — it may just mean this task wasn't large enough to break it.

**Setup:** needs only a Groq API key (`agentic_ai` in Colab secrets) and a live internet connection for the real web-search tool. No GPU needed.
firm conclusions from any single pass.

