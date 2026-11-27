### 🛡️ Guardrails for LLM Agents: four real defenses tested on the same five attacks
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abhis-28/agentic_ai_labs/blob/main/02-guardrails-and-evaluation/Guardrails_Demo.ipynb)

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

### ⚖️ Evaluating LLM Outputs: with an answer key vs without
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abhis-28/agentic_ai_labs/blob/main/02-guardrails-and-evaluation/Evaluation_Reference_Based_vs_Free.ipynb)

In normal software a test passes or fails, but "is this a good answer?" is not binary. This notebook compares **reference-based** evaluation (score against a ground-truth answer) with **reference-free** evaluation (an LLM judge scores the answer on its own), using real data (SQuAD and Dolly from Hugging Face) and real Groq calls: `openai/gpt-oss-120b` as the generator and `qwen/qwen3.8-27b` as the judge, deliberately from a different model family. It then tests the four judge modes (pointwise, pairwise, reference-guided, criteria-based) and three of the five known judge biases.

**What it showed: same generator, different ways of scoring its answers:**

| Experiment | Result |
|---|---|
| Exact-match `assert` on 2 correct answers | **Both failed** (correct answer, wrong wording) |
| Reference-based, SQuAD (one right answer), 8 questions | EM 0.875, F1 0.99 |
| Reference-based, Dolly (open-ended), 6 questions | Average F1 **0.32** |
| Reference-guided judge on the same Dolly answers | 1 of 6 "correct", 6 of 6 "correct or partially correct" |
| Faithfulness judge, answers with vs without the passage | Average score 4.5 vs 1.6; agrees with gold check on 14 of 16 |
| Wrong answers accepted by a judge with **no** reference | 4 of 6 |
| Wrong answers rejected by a judge **with** the reference | 6 of 6 |

- **Exact match fails on correct answers.** "What is the capital of Australia?" returned "The capital of Australia is **Canberra**." and `answer == expected` was `False`.
- **Reference-based scoring works only when there is one right answer.** On SQuAD it scored near-perfect, and its one miss (EM 0, F1 0.91) was the model writing "the government, the National Assembly and the Senate" against a differently worded gold answer.
- **On open-ended questions, F1 punishes valid paraphrases.** The guitar-tuning answer was correct but scored only 0.54, because the reference lists the strings highest-to-lowest and the model listed lowest-to-highest. The judge that was given the reference rated all 6 answers correct or partially correct, but only 1 as fully correct.
- **A judge without an answer key can be fooled.** For the 6 wrong closed-book answers, it accepted 4, such as "Louis Comfort Tiffany" designing the Green Dining Room (gold: Edward Burne-Jones). With the reference it caught all 6.
- **The pointwise judge didn't discriminate.** Every one of the 6 answers got 5/5 on accuracy and clarity, even the FP&A team answer the reference-guided judge called only partially correct.
- **Criteria-based judging gave the most useful feedback.** On the 6 detailed answers it passed accuracy and "answers the question" for all of them, but flagged padding on 2 (guitar tuning and the Tesla question), with a sentence saying what to cut.
- **The faithfulness judge made mistakes of its own.** It scored a correct "Edward Burne-Jones" answer 1/5, and one of the 2 disagreements with the gold check was a without-passage answer the rough word-match counted correct but the judge scored 1.
- **The position and verbosity bias tests found nothing.** The verdict never flipped when answer order was swapped (0 of 6), and the padded answer won 0 of 8 both with and without the anti-padding rubric.

**Caveats:**
- Samples are tiny (6 to 16 items), so the numbers illustrate each mechanism and are not statistics.
- The position and verbosity results do **not** show the judge is unbiased. The detailed answer simply won every pairwise comparison, so order could not matter, and the rubric could not show an effect when the padded answer never won in the first place.
- "Really correct" and "wrong answer" are labelled by a rough word-match against the gold answer, not by a human, so a few labels may be off.
- Self-preference is applied (different model family for the judge), not tested.
- Outputs vary a little between runs, and Groq's free tier may ask you to wait and re-run.

**Setup:** needs only a Groq API key, stored as `agentic_ai` in Colab secrets. Installs `langchain-groq` and `datasets`. No GPU needed.
