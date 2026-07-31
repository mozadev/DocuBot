# DocuBot AI

Ask questions about your own documents and get answers that cite where they came
from. Upload PDFs, Word files, text or Markdown; ask in plain language; get an
answer grounded in the documents, or an honest "that is not in here" when it
isn't.

Built as a take-home for the "Chat With Your Docs" brief.

---

## Contents

- [Quick start](#quick-start)
- [What it does](#what-it-does)
- [Where your data goes](#where-your-data-goes)
- [Architecture](#architecture)
- [RAG and LLM decisions](#rag-and-llm-decisions)
- [Guardrails and quality](#guardrails-and-quality)
- [Observability](#observability)
- [Key technical decisions](#key-technical-decisions)
- [Engineering standards](#engineering-standards-followed-and-skipped)
- [Productionizing this](#productionizing-this)
- [How I used AI tools](#how-i-used-ai-tools)
- [Known limitations](#known-limitations)
- [What I would do with more time](#what-i-would-do-with-more-time)

---

## Quick start

You need Docker, or Python 3.11+, and an OpenAI API key.

### Docker (recommended)

```bash
git clone https://github.com/mozadev/DocuBot-langchain-AI.git
cd DocuBot-langchain-AI

cp .env.example .env
# open .env and set OPENAI_API_KEY

docker compose up --build
```

| Service | URL |
| --- | --- |
| Web UI | http://localhost:8501 |
| API docs (Swagger) | http://localhost:8000/docs |
| Health check | http://localhost:8000/api/v1/health |

### Local Python

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env    # set OPENAI_API_KEY

# Terminal 1 - API
uvicorn api.fastapi_app:app --reload

# Terminal 2 - UI
streamlit run streamlit_app.py
```

### Tests

```bash
pip install -r requirements-dev.txt
pytest          # 87 tests, no API key and no network needed
ruff check .
```

The whole suite runs offline. Anything that would make a network call is behind
a port and gets substituted with a fake, which is the practical reason the ports
exist at all.

### Try it from the command line

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "files=@your-document.pdf"

curl -X POST http://localhost:8000/api/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the vacation policy?"}'
```

---

## What it does

- **Ingests** PDF, DOCX, TXT and Markdown. PDFs are chunked page by page so every
  citation carries a page number.
- **Reads figures.** Images embedded in a PDF are described by a vision model at
  ingest time and indexed as their own chunks, so "what does the architecture
  diagram show?" is answerable. The text layer of a PDF says nothing about its
  figures; this is the gap that closes.
- **Answers with citations**, each with the source file, page, and the retrieval
  score behind it.
- **Refuses when it should.** If retrieval comes back empty, the answer is
  blocked rather than allowed through on the model's general knowledge.
- **Traces every request.** Each answer carries a `trace_id` you can replay to
  see the retrieval scores, guardrail verdicts, cache result and latency.

---

## Where your data goes

Worth being explicit about, because "chat with your docs" means handing a
document to a system.

**Stays on the machine running this.** Uploads are written to a temp file, parsed,
and the temp file is deleted immediately. What persists is the chunk text and its
embedding in `data/vector_db/`, plus any figures extracted from a PDF as PNGs in
`data/images/`. Both directories are gitignored — an uploaded document must never
reach the repository. Under Docker they live in a named volume, not in the image.

**Leaves the machine, to OpenAI.** This is the part that matters:

- Every chunk is sent to the embeddings API at ingest.
- Every figure is sent to the vision API as base64 at ingest.
- On each question, the retrieved passages are sent in the prompt.

OpenAI's API terms state that API data is not used for training, and is retained
for up to 30 days for abuse monitoring. That is still a third party receiving the
document. **Do not upload anything you are not permitted to send to OpenAI** —
customer data, credentials, regulated personal data. For that case the swap is a
self-hosted embedding model and a local LLM, which the ports make a two-adapter
change, at a real cost in answer quality.

**Nothing is sent anywhere else.** No telemetry, no analytics, no LangSmith. The
tracer is in-process precisely so traces containing document text do not leave
the host.

---

## Architecture

```
                    Streamlit UI            REST API (FastAPI)
                         |                          |
                         +------------+-------------+
                                      |
                          +-----------v-----------+
                          |   Domain services     |   ChatService
                          |                       |   DocumentService
                          +-----------+-----------+
                                      |
              +-----------------------+------------------------+
              |                       |                        |
       +------v------+        +-------v-------+        +-------v-------+
       |   Ports     |        |   Guardrails  |        |    Tracer     |
       | (Protocols) |        | in / out      |        |  spans, cost  |
       +------+------+        +---------------+        +---------------+
              |
   +----------+----------+-------------------+
   |                     |                   |
+--v-------+      +------v------+     +------v------+
| LanceDB  |      |   OpenAI    |     |   Loaders   |
| adapter  |      |   adapter   |     | pdf/docx/txt|
+----------+      +-------------+     +-------------+
```

Hexagonal, and not for its own sake. Three things fall out of it:

1. **The tests run offline.** `FakeVectorStore` and `FakeLLM` implement the same
   Protocols the real adapters do, so 87 tests exercise real service logic
   without a key or a network.
2. **Swapping infrastructure is one file.** LanceDB to pgvector, or OpenAI to
   Bedrock, is an edit in `api/factory.py` and a new adapter. Nothing in
   `domain/` changes.
3. **The UI and the API cannot drift.** Both call the same `ChatService`, so a
   guardrail added once applies to both.

### Request path

Every question follows the same path, and each step opens a trace span:

```
question
   -> guardrail (input)      block injection before spending a token
   -> semantic cache         return a prior answer if the question matches
   -> agent graph            search -> read -> maybe search again -> answer
   -> guardrail (output)     block ungrounded answers, redact secrets
   -> cache store
answer + citations + trace_id
```

### Layout

```
domain/          entities, services, ports. No framework imports.
  guardrails.py     input and output validation
  services/         ChatService, DocumentService
  ports/            Protocols the adapters implement
adapters/        concrete implementations
  vectordb/         LanceDB
  llm/              OpenAI chat + vision
  loaders/          PDF, DOCX, text
  cache/            semantic answer cache
  observability/    tracer
agents/          LangGraph agent and its tools
api/             FastAPI routes, Streamlit UI, composition root
tests/           87 tests, all offline
```

---

## RAG and LLM decisions

### Embedding model

**Chose `text-embedding-3-small`.**

I considered `text-embedding-3-large` and open-source `bge-large` running
locally. `3-large` scores a few points higher on retrieval benchmarks and costs
6.5x more per token, for documents where the bottleneck is not embedding quality
— a handbook or a spec has fairly distinct sections, and the small model
separates them cleanly. Local `bge` removes the API dependency but adds a model
download, a GPU-or-slow decision, and a much heavier container, all to solve a
cost problem that does not exist at this scale.

The honest version: at 1536 dimensions and this corpus size, retrieval quality is
limited by chunking, not by the embedding model. I would rather spend the
complexity budget there.

### Chunking

**Chose recursive character splitting, 1000 chars with 200 overlap, split per
page for PDFs.**

The size came from what a chunk has to do: hold enough of a passage to answer a
question on its own, while staying small enough that four of them fit in a prompt
without burying the model. 1000 characters is roughly a paragraph or two.

The 200-char overlap is the part that matters most. Without overlap, a fact that
straddles a boundary is in neither chunk in a usable form — "employees accrue 20
days" ends one chunk and "which may be carried over" starts the next, and neither
retrieves well for a question about carry-over. 20% is the usual rule of thumb and
I did not have an evaluation set to tune it against, which I would want before
claiming a better number.

Splitting **per page** rather than over the whole document was a deliberate
choice: it means every chunk knows its page, so a citation can say "page 7" and a
user can go verify it. The cost is that a passage spanning a page break gets cut.
For a verification-driven product that is the right side of the trade.

Separators are ordered so Markdown headings break first (`\n## `), then
paragraphs, then sentences. A heading is often the only place a section's subject
is named, so keeping it attached to its body matters.

**What I did not do:** semantic chunking, where you embed sentences and split at
similarity troughs. It is better in principle and I have no evaluation harness to
prove it is better here, so it would have been complexity on faith.

### Vector store

**Chose LanceDB.**

Considered pgvector, Qdrant and Pinecone. The deciding factor was that LanceDB is
*embedded* — no server, no second container, the index is a directory. That is
what makes `git clone && docker compose up` actually work for someone reviewing
this, and a reviewable project is worth more here than marginal recall.

It is also the decision I would revisit first in production, and the port makes
that cheap. See [Productionizing](#productionizing-this).

I dropped the `langchain-community` LanceDB wrapper and talk to LanceDB directly.
The wrapper is now sunset, and it hid the two things worth controlling: how
metadata is stored (flat columns, so filtering by filename or page is a plain SQL
predicate later) and how distance becomes the score a user sees.

### LLM

**Chose `gpt-4o-mini` for answering, `gpt-4o` for reading figures.**

Two models, split by how often each runs. Answering happens on every turn;
describing a figure happens once per document at ingest. Paying more for accuracy
at ingest is the cheap side of that trade, and a misread chart is permanently
wrong in the index, while a mediocre answer can be re-asked.

`gpt-4o-mini` is enough for answering because the hard part — finding the right
passage — has already happened by the time the model runs. It is being asked to
read four paragraphs and answer from them, not to know things.

Temperature 0.2, not 0. At exactly 0 the model tends to parrot the source text
verbatim, awkward phrasing and all. A little slack lets it write a sentence while
staying grounded.

### Orchestration

**Chose LangGraph over a plain LCEL chain.**

A fixed retrieve-then-generate chain is simpler and I nearly used it. The agent
loop earns its extra round-trip in two specific cases:

- The first search returns nothing useful and the model reformulates and tries
  again. This happens more than I expected — users ask "what's the PTO policy"
  about a document that says "vacation".
- A turn that needs no retrieval at all ("what did I just ask you?") skips it,
  instead of burning an embedding call and stuffing four irrelevant chunks into
  the prompt.

The cost is one extra LLM call on tool-using turns and a graph that is harder to
reason about than a chain. I think it is worth it; I would not have bothered for
a single-shot QA endpoint.

### Prompt and context management

The system prompt is in `agents/graph.py`, written as a short numbered procedure
rather than prose. Three things in it are load-bearing:

- **"If the documents do not contain the answer, that is the answer."** Stated
  positively, as a correct outcome rather than a failure. Models are strongly
  inclined to be helpful, and framing refusal as failure makes them fill gaps.
- **"Search again if the first search misses."** Without this the model accepts
  the first empty result and gives up.
- **"Reply in the language the user wrote in."** Rather than pinning a language.

Context is bounded in three places: 4 chunks per search, 1000 characters shown per
chunk, and a 6-turn rolling history window. The history window is the one worth
explaining — full history grows the prompt without bound and, worse, old turns
start competing with retrieved passages for the model's attention. Six turns
covers the follow-ups people actually ask.

---

## Guardrails and quality

Two checkpoints, because input risk and output risk are different problems.

**Input** (`check_input`) runs before any token is spent. It blocks prompt
injection — instruction override, prompt exfiltration, role reassignment — and
oversized input. Injection patterns are matched against the *question only*,
never document content: a PDF that happens to contain the words "ignore previous
instructions" is data, not an attack.

The tests include the case I care about more than the attacks: legitimate
questions that contain trigger words. "What are the onboarding instructions?" must
not be blocked. A guardrail that blocks real questions gets turned off.

**Output** (`check_output`) runs before the answer is returned, and the grounding
check is the one that makes this a document assistant rather than a chatbot with
documents attached:

- If retrieval returned **nothing** and the model answered anyway, the answer is
  blocked and replaced with a refusal. It came from parametric memory, and in
  this product that is a wrong answer even when the fact is true.
- An honest refusal with no sources passes — that is the correct behaviour.
- Weak grounding (top score below 0.25) passes with a warning, surfaced in the UI
  as a low-confidence badge.
- Emails, card numbers and API-key-shaped strings are redacted.

Both are rule-based, not an LLM judge. A judge would catch more nuance, but it
doubles latency and cost on every turn and can itself be talked out of a verdict.
Rules are cheap, deterministic and testable. The judge is the upgrade path, not
the starting point.

### Quality controls

- 87 tests, offline, covering guardrails, ingestion, retrieval, the agent's
  citation extraction, the service composition and the HTTP contract.
- `ruff` in CI with bugbear and blind-except rules on.
- Citations are rebuilt from what retrieval **actually returned**, not from what
  the model claims it used, so a hallucinated filename cannot reach the user.
- Confidence is surfaced everywhere rather than hidden, so a weak answer looks
  weak.

The gap I am most aware of: there is **no retrieval evaluation set**. I have no
recall@k number for this system. See [what I would do with more
time](#what-i-would-do-with-more-time).

---

## Observability

Every request opens a trace; every step opens a span. `GET
/api/v1/observability/traces/{trace_id}` returns the full breakdown, and the UI's
Traces tab renders the same data.

This is not decoration — it found a real bug during development. A grounded
question was coming back as "not in these documents" even though the document
was indexed. The trace showed it immediately:

```
guardrail_input   {"passed": true}
cache_lookup      {"hit": false}
agent_graph       {"answer_chars": 134, "sources": 0}
vector_search     {"results": 0, "top_score": 0.0}   <- here
vector_search     {"results": 0, "top_score": 0.0}
```

Retrieval was returning nothing while indexing reported success. The cause:
`lancedb` 0.36 returns a paginated response object from `list_tables()` where
older versions returned a list, so the table-existence check silently evaluated
false and every search short-circuited to empty. Without the span I would have
been reading the prompt, suspecting the model.

That bug is now covered by a regression test in `tests/test_vector_store.py`, and
the reason it was possible at all is in
[how I used AI tools](#how-i-used-ai-tools).

Also exposed: `/observability/analytics` (aggregate latency, cost, error rate),
`/cache/stats`, `/rate-limit/usage`.

Traces are in-memory and bounded at 500. That is a deliberate scope choice, not
an oversight — see below.

---

## Key technical decisions

**Cut the scope hard.** This repository previously contained a marketing content
generator: 40 endpoints, SEO tooling, DALL-E image generation, LinkedIn Ads,
brand personas. All of it is gone. The brief says a solid basic solution beats an
over-engineered one, and I agree with it strongly enough to delete working code.
What is left is one thing done properly. The history is in git.

**Pinned every dependency.** The previous revision used `>=` ranges, and
`mcp>=1.0.0` resolved to a 2.x release that had moved a module — a clean
`pip install -r requirements.txt` no longer imported. Reproducible beats current
for anything someone else has to run. This is also why CI builds the Docker image
on every push.

**Session-scoped state.** Conversation history and cache are keyed by
`X-Session-ID`. The previous version had one global history shared by every
caller. This is scoping, not security — there is no auth, and the header is
trusted. That is a deliberate, stated limitation rather than an accident.

**Citations carry the original filename.** Uploads land in temp files, and the
naive implementation cited `tmp9rdv1gq9.pdf`. The original name is threaded
through to the loader. Small thing, but a citation you cannot recognise is not a
citation.

**Streaming bypasses the output guardrail, on purpose.** `/stream/chat` emits
tokens as they are produced, which means you cannot retract one. The guardrail
runs on the input side only, and the UI uses the non-streaming path where the
grounding verdict matters. Documented rather than quietly ignored.

**Health checks touch nothing.** `/health` does not call OpenAI or the vector
store. A liveness probe that depends on a third party will restart healthy
containers during their outage. Dependency state lives in `/status`.

---

## Engineering standards followed and skipped

**Followed**

- Hexagonal architecture with Protocol-based ports, and it pays for itself in the
  test suite rather than being decoration.
- 87 tests, all offline and deterministic. Tests named as behaviour
  (`test_ungrounded_answer_is_replaced_with_a_refusal`) so a failure reads as a
  broken requirement.
- CI on every push: lint, test, Docker build.
- Pinned dependencies, multi-stage Docker build, non-root container user,
  layer-cached dependency install.
- Config entirely from environment, validated by Pydantic at import so a missing
  key fails at startup with a clear message.
- Structured tracing wired through the real request path.
- Comments explain *why*, not *what*. Where a number is arbitrary I said so.

**Skipped, knowingly**

- **Authentication.** No auth on any endpoint. For a reviewable take-home the
  setup cost outweighs the signal, but it is the first thing to add.
- **Type checking.** No mypy. Ruff catches a good share; a strict-mode pass would
  need annotations I did not add everywhere.
- **Integration tests against real OpenAI.** The suite is fully mocked. A small
  smoke suite behind the `integration` marker would catch API contract drift.
- **Property-based tests.** The chunking logic would suit Hypothesis well.
- **Pre-commit hooks.** CI covers it; local hooks are a convenience I skipped.
- **Frontend build.** Streamlit, not React. Discussed under limitations.
- **Migrations for the vector store.** Changing the schema today means
  re-indexing.

---

## Productionizing this

What is here is a single container with a local index. Below is what changes to
make it a real deployment, roughly in the order I would do it.

### The blocker: state is local

Three things live in process memory or on local disk: the LanceDB index, the
semantic cache, and the traces. Any of them makes horizontal scaling wrong, not
just suboptimal — two replicas would each have half the cache, their own trace
history, and with a local volume, their own index.

| Component | Now | Production |
| --- | --- | --- |
| Vector store | LanceDB on local disk | **pgvector** on RDS/Cloud SQL, or Qdrant. Shared, backed up, transactional with your metadata. |
| Cache | In-process dict | **Redis** (ElastiCache / Memorystore). Shared hit rate, survives deploys. |
| Traces | In-memory, 500 max | **OpenTelemetry** to CloudWatch / Cloud Trace / Datadog. The span model here maps onto OTLP deliberately. |
| Sessions | In-process dict | Redis or DynamoDB, with a TTL. |
| Rate limiting | Per-process counter | Redis token bucket, or the API gateway's own limiter. |

pgvector is my default recommendation: at this corpus size it is fast enough, and
one less system to operate is worth real money. Qdrant if the corpus reaches tens
of millions of chunks.

### Deployment shape (AWS as the example)

```
Route53 -> CloudFront -> ALB -> ECS Fargate (API, 2+ tasks, autoscaled on ALB
                                             request count)
                                    |
                +-------------------+--------------------+
                |                   |                    |
          RDS Postgres         ElastiCache            S3 (uploads,
          + pgvector             Redis                extracted figures)
                |
          Secrets Manager (OPENAI_API_KEY, DB credentials)
```

Equivalents: Cloud Run + Cloud SQL + Memorystore on GCP; Container Apps +
Postgres Flexible Server + Azure Cache on Azure.

**Ingestion must move off the request path.** A 200-page PDF with figures takes
minutes and dozens of vision calls. Today that blocks an HTTP request. In
production: upload to S3, enqueue to SQS, process in a worker, and let the client
poll a job status endpoint. This is the single biggest change on the list.

### Also required

- **Auth.** OIDC or API keys at the gateway, and document access scoped per
  tenant — which means a tenant column on every chunk and a filter on every
  query, not just a header.
- **Secrets** from Secrets Manager, never environment variables in a task
  definition.
- **Cost controls.** Per-tenant token budgets, alerts on daily spend, and a hard
  ceiling. An unbounded LLM endpoint is an unbounded bill.
- **Structured JSON logs** with a request id propagated through every span.
- **A retrieval evaluation set in CI**, so a prompt or chunking change that hurts
  recall fails the build instead of shipping.
- **PII handling.** Uploaded documents may contain personal data: encryption at
  rest, a retention policy, and a real delete path.
- **Backups** of the vector store, with a restore drill that has actually been
  run.

---

## How I used AI tools

I used Claude Code throughout. Being specific about how, since generic answers
here are not useful.

**Where it did the most good**

- *Mechanical breadth.* Translating the codebase from Spanish, splitting oversized
  modules, applying a consistent pattern across a dozen route handlers. Work that
  is tedious and where a human's attention drifts.
- *Filling in test cases around a spec I set.* I decided what the guardrails must
  guarantee — "a legitimate question containing the word 'instructions' must not
  be blocked" — and had it enumerate cases against that. It is good at breadth
  once the invariant is stated, and bad at deciding the invariant.
- *Rubber-ducking trade-offs.* Arguing for pgvector against Qdrant surfaced
  considerations faster than reading three docs sites.

**Where I did not trust it**

- *Architecture and scope.* The decision to delete the entire marketing subsystem
  is the highest-value change in this submission and it is a judgment call about
  what the reader values. No assistant was going to tell me to delete working
  code.
- *The numbers.* Chunk size, overlap, the 0.25 grounding threshold, the 0.92 cache
  similarity threshold. It will produce a confident value for each. I set them
  from reasoning I can defend, and where the reasoning is thin I said so above
  rather than dressing it up.
- *Prompt wording.* The system prompt went through several hand-edits. "If the
  documents do not contain the answer, that is the answer" is phrasing I arrived
  at after watching the model hedge and fill gaps with generated text.

**The lesson that cost me the most time**

The LanceDB bug in the [Observability](#observability) section is the honest
example. The generated adapter code used `table_name in db.list_tables()`, which
is correct against the library's documented behaviour and wrong against the
version in `requirements.txt`, where that method returns a paginated object. It
looked right, it passed review by eye, and indexing reported success — the
failure only showed up two layers away as "no results found."

This is the characteristic failure mode: not obviously-wrong code, but code that
is plausible against a *slightly different version* of reality. It is invisible to
reading and obvious to running. That is why the fix was not just a patch but a
regression test that exercises the real database, and why I write tests against
real infrastructure wherever it is cheap enough to do so.

**My rules, stated plainly**

- Do: use it for breadth, translation, mechanical refactors, and test enumeration
  once I have set the invariant.
- Do: run everything. Generated code that has not been executed is a hypothesis.
- Do: make it explain a trade-off, then decide myself.
- Don't: let it choose architecture, scope, or thresholds.
- Don't: accept a comment that restates the code. Comments earn their place by
  explaining a decision.
- Don't: let it write the parts of this README that are supposed to be my
  reasoning. The sections above are mine; that is the whole point of them.

---

## Known limitations

Things I know are wrong or missing, rather than things I hope you do not notice.

- **No authentication.** Every endpoint is open. `X-Session-ID` is trusted and
  scopes state; it secures nothing.
- **Single-tenant index.** All documents share one collection. Anyone can query
  anything that has been uploaded.
- **Ingestion is synchronous.** A large PDF with many figures will hold an HTTP
  request open for minutes and may time out behind a proxy.
- **No OCR.** A scanned PDF with no text layer indexes zero text chunks. The
  loader logs a warning; it does not tell the user in the UI, which it should.
- **Retrieval is dense-only.** No BM25, no hybrid, no reranking. Exact-match
  queries — an error code, a part number — are the weak spot, because embeddings
  are bad at rare literal tokens.
- **No evaluation set.** I cannot give you a recall@k number for this system.
- **In-memory everything** for cache, traces and sessions. Lost on restart, wrong
  across replicas.
- **Streaming skips the output guardrail**, as described above.
- **Confidence is the top retrieval score**, which measures whether we found
  something relevant, not whether the answer is right. It is a useful signal
  presented honestly, not a correctness probability.
- **The UI is Streamlit.** Good enough to demonstrate the product, and not what I
  would ship. A React frontend against the existing REST API would be the change;
  the API is already the real interface.

---

## What I would do with more time

In priority order, which is itself the answer to what I think matters.

1. **An evaluation set.** 40 to 50 question/answer pairs over a fixed corpus,
   measuring recall@k, citation correctness, and refusal accuracy on questions
   the documents genuinely do not answer. Run it in CI. Everything below this is
   guesswork without it, including whether any of the other changes help.

2. **Hybrid retrieval with reranking.** BM25 alongside dense retrieval, fused,
   then a cross-encoder reranker over the top 20. Dense-only retrieval is the
   clearest weakness, and this is the standard fix — but I want number 1 first so
   I can prove it helped rather than assume it.

3. **Asynchronous ingestion.** Queue and worker, with job status polling. Removes
   the timeout ceiling on large documents.

4. **Auth and real multi-tenancy.** A tenant column on every chunk, filtered on
   every query, behind OIDC.

5. **Move state to Redis and pgvector.** Everything in the productionizing table.

6. **A React frontend.** Streaming answers with token-by-token rendering, inline
   citation highlighting against a PDF viewer, so a user can click a citation and
   see the highlighted passage in the source. That is the feature that would most
   change how much people trust the output.

7. **An LLM-as-judge guardrail** on top of the rules, for the nuance rules cannot
   catch — an answer that cites a real passage but misreads it.

8. **OCR** for scanned PDFs, and telling the user in the UI when a document
   indexed no text.

---

## License

MIT. See [LICENSE](LICENSE).
