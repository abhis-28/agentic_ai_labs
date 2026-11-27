
# Working Memory in Agentic AI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abhis-28/agentic_ai_labs/blob/main/04-agent-memory/Working_Memory_Demo.ipynb)

Working memory is everything inside the LLM's context window: the system prompt, the conversation history and the tool outputs. This notebook builds a LangGraph agent (Groq `openai/gpt-oss-120b` + Tavily web search) and shows, with real LLM calls only, how the window fills up and what happens when you shrink it. Token counts come from the API's own `usage_metadata`. The "context window" is a pretend 2,000-token budget so the problem shows up on a free tier.

## What it showed

- **The LLM is stateless.** Told "My name is Kartik" in one call, a separate call could not answer "What is my name?". Resending the messages (107 input tokens) fixed it.
- **Tool outputs fill the window.** Memory grew 147 → 404 → 850 → 2,327 → 3,125 tokens over five turns, passing the 2,000 budget at turn 4 (the Mumbai weather search).
- **Truncation fails silently.** The user's turn-1 rule was "never mention Samsung". With the full history (3,291 tokens), the top-5 phone brands answer followed it. Keeping only the last 2 turns (1,867 tokens) dropped the rule, and the model put Samsung at #1 with no warning.
- **Summarisation depends on the prompt.** Both summaries cut memory to 552 and 681 tokens. The generic summary kept only "never mention a specific brand", and the final answer was a refusal ("I'm sorry, but I can't provide that information."). The check still scored it as not breaking the rule. The constraint-aware summary kept the Samsung rule and returned a usable table, but it hinted at the hidden brand ("The omitted brand typically occupies the leading position").
- **Selective retention was cheapest.** An LLM scorer kept the rule turn (score 10) and dropped the four unrelated search turns (score 0), so only 173 tokens were sent and the rule held. Its answer put Apple (2.6%) at rank 5 instead of OnePlus, because the brand removed by the rule left a gap that the model filled with a lower-ranked brand.
- **Replace corrected info, don't stack it.** Without search the model said Joe Biden was president. After a correction the clean memory (2 messages, 298 tokens) was about 70% smaller than the dirty one (10 messages, 1,006 tokens). Both produced the same correct greeting for President Trump.

| Approach | Tokens sent | Rule broken? |
|---|---|---|
| Full history | 3,291 | No |
| Truncation (last 2 turns) | 1,867 | **Yes** |
| Summary (generic) | 552 | No (but refused to answer) |
| Summary (constraint-aware) | 681 | No |
| Selective retention | 173 | No |

## Caveats

- Outputs vary between runs, so your numbers and wording will differ a little.
- The rule check is a substring match on "samsung". It cannot detect hints, refusals or wrong rankings, so "No" does not mean the answer was good.
- Search results can be stale. The ISRO answer covered late 2024 to mid 2025, and the model mislabeled the weekday in the Mumbai forecast.
- Selective retention is only as good as its scorer. A wrong low score permanently drops needed information.
- Single run, single rule, one model, so these are demonstrations, not benchmarks.

  # Episodic Memory in Agentic AI

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abhis-28/agentic_ai_labs/blob/main/04-agent-memory/Episodic_Memory_Demo.ipynb)

Episodic memory is a persistent record of what happened in past sessions. It lives outside the LLM (here, a JSON file) and only the episodes retrieved for the current question are put back into the context window. This notebook builds a LangGraph agent (Groq `openai/gpt-oss-20b`) that saves each finished session as an episode (summary, key events, outcomes, timestamp, embedding) and compares two ways to retrieve them: chronological ("last N sessions") and semantic (cosine similarity with `all-MiniLM-L6-v2`). All assistant replies, searches and summaries are real LLM calls. The user's messages are scripted, and sessions run minutes apart rather than days.

## What it showed

- **No memory, no continuity.** A returning user asking "continue what we were doing last time" got "Could you remind me what we were working on?". Asked about the rule from the EV-report session, the agent said it saw no such rule (143 tokens sent).
- **Three sessions became three short episodes.** An EV-report session, a pipeline `KeyError` fix and a Jaipur trip plan were each saved as an episode with a summary, key events and outcomes, instead of the raw transcript.
- **Chronological retrieval won on recency and lost on older sessions.** For the vague "continue what we were doing", the last-1 episode gave the right answer (Bani Park, 293 tokens). For the EV rule, the last 2 episodes (Pipeline error and Jaipur trip) did not contain the EV session, so the agent could not recall the rule (408 tokens).
- **Semantic retrieval won on specific questions and lost on vague ones.** For the EV rule question it ranked the EV session first (similarity 0.55 vs 0.15) and answered "exclude any discussion of charging infrastructure" (269 tokens). For the vague question, the scores were low and close (0.19 vs 0.15 vs 0.10). It retrieved the EV session, not the Jaipur one, and the answer missed Bani Park.
- **Key events and outcomes were usable.** It recalled the "concise" preference (similarity 0.23 vs 0.15, a thin margin) and the pipeline fix (`customer_id` renamed to `cust_id`, similarity 0.68), each with about 260-270 tokens of memory.
- **Summaries lose detail.** Asked to list every area suggested for staying in Jaipur, the raw transcript (3,074 tokens) listed five (Sindhi Camp, Bani Park, MI Road, Panchpura, Kishanpura). The stored episode (297 tokens) returned only Bani Park.

| Approach | Tokens sent | Expected fact in answer? |
|---|---|---|
| No memory (EV question) | 143 | No |
| Chronological, last 1 (vague) | 293 | Yes |
| Chronological, last 2 (EV) | 408 | No |
| Semantic, top 1 (EV) | 269 | Yes |
| Semantic, top 1 (vague) | 267 | No |
| Semantic, top 1 (event) | 269 | Yes |
| Semantic, top 1 (outcome) | 260 | Yes |

## Caveats

- User messages are scripted, and the keyword check is only a quick signal.
- Sessions run minutes apart, not days. The mechanism is the same, but nothing here tests long-term drift.
- The web search is flaky from Colab (it showed `DNSError` retries), and some search-derived claims in the agent's answers ("2026 prices") were not checked.
- Episode text is written by an LLM, so outcomes in an episode are not guaranteed accurate.
- Only 3 episodes, one run and one embedding model, so this demonstrates the mechanism and is not a benchmark.

## Setup

Colab secret: `agentic_ai` (Groq API key). Runtime → Run all. Add `episodes.json` to `.gitignore`.

## Semantic Memory in Agentic AI
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abhis-28/agentic_ai_labs/blob/main/04-agent-memory/Semantic_Memory_Demo.ipynb)

Semantic memory is knowledge an agent can reuse across conversations. This notebook compares two kinds, using real Wikipedia pages, Groq `gpt-oss-20b`, `all-MiniLM-L6-v2` embeddings and Mem0 with a local Qdrant store.

**World knowledge (shared documents): RAG works**
- Without RAG, the model answered "Rakesh Sharma" for the Indian astronaut on Axiom Mission 4. That is wrong, and it was stated with confidence.
- With the top 3 retrieved chunks, it correctly answered Shubhanshu Shukla and the ISS. The question and the answering chunk are semantically close.

**User knowledge (personal facts): plain RAG falls short**
- A scripted 8-message chat with a user, Riya, is the test data. One message ("never written Python") is what matters for the query "Give me code for Transformers".
- Similarity search ranked that message 5th of 8, so the top-2 retrieval skipped it.

**Mem0: Add, Search, Update, Delete**
- A short LLM prompt turned the 8 messages into 8 concise facts. Mem0 stored them with `infer=False`, to fit Groq's 8,000 tokens-per-minute limit.
- Search returned only one memory (about transformers). The Python fact was still not retrieved, and the model's reply came back empty in that step.
- Update (delete the old fact, add the new one) and Delete (removing "vegetarian") both worked on the stored memories. Both are done by our code, not Mem0's automatic pipeline.

**Takeaway:** distilled facts give a clean memory that can be updated or deleted. Retrieval is still similarity-based, so the fact that matters may not be retrieved. This run does not benchmark Mem0's built-in extraction.

## Setup

Colab secrets: `agentic_ai` (Groq API key) and `tavily_api_key` (free key from tavily.com). Then Runtime → Run all.
