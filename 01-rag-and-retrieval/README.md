# RAG and Advanced Retrieval

This directory contains practical implementations of RAG,
document chunking, sparse retrieval, advanced retrieval,
Self-RAG, and RAG evaluation.


### Late Chunking: embed first, chunk later
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<abhis-28>/<
agentic_ai_labs>/blob/main/Late_Chunking_Demo.ipynb)

Normal chunking splits a document and embeds each chunk alone, so a sentence like
"With 3.7 million inhabitants, it has the highest population..." loses what "it" refers to.
Late chunking passes the **whole document** through a long-context embedding model first,
then averages the token embeddings of each chunk, so every chunk keeps the context of the
document around it.

**What the notebook does:** loads the Wikipedia intro for Berlin (27 sentence chunks, 709 tokens),
builds normal and late-chunking embeddings with `nomic-ai/modernbert-embed-base`, and runs the
same RAG question through both with `gpt-oss-20b` on Groq.

**What it showed:** for "What is the population of Berlin?", the sentence giving the city's
3.7 million figure moved from rank 11 (normal) to rank 2 (late). Late chunking retrieved it and
answered 3.7 million, while normal chunking missed it and answered with the urban-area figure,
4.5 million.

**Caveats:** one document and one question, so this is an illustration, not a benchmark. It needs
a long-context embedding model that returns token-level embeddings and was trained with mean pooling.

**Setup:** add your Groq key to Colab secrets as `agentic_ai`.


### Sparse Retrieval (BM25): match exact words, no embeddings
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<abhis-28>/<
agentic_ai_labs>/blob/main/Sparse_Retrieval_BM25_Custom_Text.ipynb)

Sparse retrieval matches the **exact words** of a query against the words in each chunk. BM25
scores a chunk higher when it contains words that are **rare** across the text (high IDF weight),
and it needs no embedding model or GPU.

**What the notebook does:** splits a short three-topic text (photosynthesis, the French Revolution,
machine learning) into 6 chunks, builds a `BM25Okapi` index, and runs RAG with `gpt-oss-20b` on Groq.
A table shows, for every query word, how many chunks contain it and its BM25 weight.

**What it showed:**
- **Exact terms work well.** For "What absorbs light in chloroplasts during photosynthesis?" the
  chunk with those rare words ranked first (score 4.15). "When was Louis XVI executed?" retrieved
  the right chunk and the LLM answered 1793. Words in many chunks, like "in", get weight 0.
- **Synonyms get no credit.** In "Which French monarch was beheaded in 1793?", the words `monarch`
  and `beheaded` scored zero. In this tiny text, though, the shared words `french` and `1793` still
  pulled the right chunk to the top, so the answer was correct. BM25 only fails when a query
  shares no useful words with the text.

**Caveats:** a 6-chunk toy corpus, so retrieval is easy. It illustrates how BM25 scores words,
not how it performs on real data.

**Setup:** add your Groq key to Colab secrets as `agentic_ai`.

### SPLADE, ColBERT and Ensemble Retrieval
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<abhis-28>/<
agentic_ai_labs>/blob/main/SPLADE_ColBERT_Ensemble_LangChain_Demo.ipynb)

Three retrieval techniques compared on a short text about AI in healthcare (6 sentence chunks),
each run through a full RAG flow (retrieve, augment, generate) with `gpt-oss-20b` on Groq:
- **SPLADE** is learned sparse retrieval: keyword search where a neural model first expands each word with related words.
- **ColBERT** keeps one embedding per token and scores chunks by token-level matching (MaxSim).
- **Ensemble** runs SPLADE, dense and ColBERT and merges their rankings with Reciprocal Rank Fusion.

Three test questions: an exact term, a paraphrase, and a question sharing almost no words with the answer.
The notebook measures the **rank of the right chunk** (1 = best of 6).

| Question | SPLADE | Dense | ColBERT | Ensemble |
|---|---|---|---|---|
| Exact term ("What are CNNs used for?") | 1 | 1 | 1 | 1 |
| Paraphrase ("...obstacles stop hospitals...") | 2 | 1 | 1 | 1 |
| No shared words ("...computers understand what doctors write...") | 3 | 1 | 1 | 1 |

**What it showed:**
- SPLADE expanded an 8-word question into 52 weighted terms (adding words like "barriers" and
  "prevent"), and chunk vectors grew from about 16 words to about 106 terms. But it still ranked
  the right chunk 2nd and 3rd on the harder questions, and answered "I don't know" on the last one.
- ColBERT stored 123 vectors for 6 chunks (dense would store 6), and ranked the right chunk 1st every time.
- The ensemble also ranked the right chunk 1st every time, so it was not dragged down by SPLADE's misses.

**Caveats:** 6 chunks and 3 questions, so even the weakest method here still finds the right chunk
in its top 3. Dense and ColBERT were already perfect, so this demo cannot show the ensemble beating its
best member. It illustrates how each method works, not how they compare on real data.

**Setup:** add your Groq key to Colab secrets as `agentic_ai`. Runs on CPU. It downloads
`prithivida/Splade_PP_en_v1`, `colbert-ir/colbertv2.0` and `sentence-transformers/all-MiniLM-L6-v2`.

### Self-RAG: a model that decides when to retrieve and critiques its own answers
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<abhis-28>/<
agentic_ai_labs>/blob/main/SelfRAG_4bit_Demo.ipynb)

Runs the real Self-RAG model (`selfrag/selfrag_llama2_7b`, loaded in 4-bit on a free Colab T4 GPU).
It writes special **reflection tokens** such as `[No Retrieval]`, `[Relevant]`, `[Fully supported]`
and `[Utility:5]`, which the notebook uses to make four decisions:
1. **Retrieve or not?** The probability of `[Retrieval]` vs `[No Retrieval]` as the model's first token.
2. **Is each retrieved chunk relevant?**
3. **Is the answer supported by the chunk?**
4. **Is the answer useful?** (1 to 5)

The chunks come from a short AI-in-healthcare text (6 sentence chunks) searched with
`all-MiniLM-L6-v2` embeddings. The notebook picks the best chunk with a simple scoring rule of its own
and warns when no chunk supports the answer.

**What it showed (5 questions):**
- The reflection tokens exist in the model's vocabulary, and it writes them in its output.
- **No retrieval needed:** "What is 2+2?" (probability 0.00) was answered directly, and so was a poem
  about autumn (0.13).
- **Answer in the text:** for "Which technique helps extract information from clinical notes?" (retrieve
  probability 0.90), the right chunk scored highest: relevance 1.00, fully supported, utility 5.
- **Partial support:** for "What are CNNs used for?", all three chunks scored the same, and the answer
  added "object detection" and "image segmentation", which are not in the text. The model marked it only
  partially supported.
- **Answer not in the text:** for "Who won the FIFA World Cup in 2022?", no chunk supported the answer,
  so the notebook warned that it came from the model's own knowledge. Utility was still 5, so utility
  measures usefulness, not whether the answer is grounded.

**Caveats:** 5 questions and 6 chunks. The 0.2 retrieval threshold and the scoring formula are
choices made in this notebook, and 4-bit quantization may change the model's behaviour.

**Setup:** needs a **T4 GPU** (Runtime → Change runtime type → T4 GPU). No API key needed. The first
run downloads the roughly 13 GB model, which takes several minutes.

Model from the Self-RAG paper (Asai et al., 2023).

## RAG Assessment with RAGAS
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<abhis-28>/<
agentic_ai_labs>/blob/main/RAG_Assessment_RAGAS_Demo.ipynb)

A RAG system can fail in different ways: noisy retrieval, missing information, hallucination or an off-topic answer. This notebook builds a small RAG system over a healthcare text (6 sentence chunks, `all-MiniLM-L6-v2` embeddings). It writes the four RAGAS metrics from scratch, using Groq `gpt-oss-20b` inside the RAG system and a bigger `gpt-oss-120b` as the judge. It then breaks the system on purpose in five ways, using 3 questions with ground-truth answers.

| Scenario | Faithfulness | Answer relevance | Context precision | Context recall |
|---|---|---|---|---|
| Healthy RAG (k=2) | 0.87 | 0.66 | 0.50 | 0.83 |
| Low recall (k=1) | 0.83 | 0.74 | 1.00 | 0.83 |
| Low precision (k=6) | 1.00 | 0.73 | 0.22 | 1.00 |
| Hallucination | 0.50 | 0.65 | 0.50 | 0.83 |
| Off-topic answer | 0.33 | 0.52 | 0.50 | 0.83 |

**What the run shows**
- **Context precision** responded as expected: 1.00 with one chunk, 0.22 with all six.
- **Faithfulness** dropped when the model was allowed to add its own knowledge (0.50 vs 0.87). In the one-metric demo, the judge flagged 6 of 8 claims as unsupported.

**What it shows less clearly**
- **Context recall** did not drop in the "low recall" scenario (0.83, same as healthy RAG). With only 3 questions, k=1 happened to retrieve what was needed for most of them.
- **Answer relevance** barely separated the scenarios (0.52 to 0.74).
- The off-topic prompt lowered faithfulness more than relevance, because the overview prompt also brings in claims outside the context.
- **The judge is noisy.** Running the identical notebook twice gave different per-question scores (for example, hallucination Q1 faithfulness was 0.50 in one run and 0.25 in the other), though the scenario averages stayed close. Treat single scores with caution, and read trends across scenarios.

**Takeaway:** the metrics point to where a RAG system is failing, but they depend on the judge model and on the test questions.
