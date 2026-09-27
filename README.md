# MediTrack AI — Complete MVP

**AI Voice Agent for Pharmacy Enquiry & Order Automation**
Stage II, Week 4: Complete MVP Demonstration, Validation & Hardening

A customer phones the pharmacy, asks about a medicine, and places an order —
answered entirely by an AI agent grounded in the pharmacy's real formulary and
stock, with anything clinical or uncertain handed to the pharmacist.

```
Voice → Speech-to-Text → Intent + Entities → Knowledge Base → Order Workflow → Spoken Reply
                                                    ↓
                                         Pharmacist escalation queue
```

## Status

| Week | Deliverable | Status |
|---|---|---|
| 1 | MVP foundation & voice agent prototype | Complete |
| 2 | Pharmacy knowledge base & medicine query handling | Complete |
| 3 | Order creation & pharmacy workflow integration | Complete |
| 4 | Complete MVP demonstration & validation | Complete |
| 4b | Hardening: input validation, auth, webhook verification, recovery | Complete |

Architecture, data model and design decisions: **[ARCHITECTURE.md](ARCHITECTURE.md)**

## What the MVP does

- **Answers the phone** — Twilio voice webhooks, or a local microphone for demos.
- **Understands mangled drug names** — phonetic search resolves `amoxy sillin`,
  `paracetemol`, `ventalin`, `sitirizeen` to the right record.
- **Answers from real data** — availability, price, pack size, prescription
  status, storage and shelf location for 28 medicines. Never invents a fact.
- **Takes orders over the phone** — collects medicine, quantity, name, phone,
  pickup or delivery and address across turns, then writes an order record.
- **Checks inventory properly** — availability is on-hand minus everything
  already reserved, so two callers cannot take the same last pack.
- **Escalates what it must not decide** — dosage questions, prescription-only
  items, cold-chain items, stock shortfalls and requests for a human all go to
  a prioritised pharmacist queue.
- **Answers status callbacks** — "is my order one zero zero one ready?" is
  looked up by order number or phone.
- **Degrades safely** — with no LLM key, a failed API call, a missing
  formulary, silence on the line, or an unrecognised medicine, it still gives a
  correct spoken answer (or a clear `/health` status) rather than crashing.
- **Refuses what it should** — a zero, negative or absurd quantity is re-asked
  rather than persisted; the staff API needs an `X-API-Key`; voice webhooks
  need a valid Twilio signature.

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then add your API key
```

Set `LLM_PROVIDER` to `openai` or `anthropic` and supply the matching key.
Whisper needs `OPENAI_API_KEY`. **Every demo below runs without any key** —
replies fall back to the knowledge base.

Also set `ADMIN_API_KEY` in `.env` — the staff endpoints (`/orders`,
`/escalations`, `/kb/*`, `/query/analyze`) refuse every request with a 503
until it is set, by design. Generate one with:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

## Demonstration

```powershell
# Week 4 - the complete journey: order → pharmacist review → status callback
python -m tests.test_pipeline_local --full-demo

# Week 3 - order taking, and the four escalation paths
python -m tests.test_pipeline_local --order-demo
python -m tests.test_pipeline_local --escalation-demo

# Week 2 - knowledge base grounding
python -m tests.test_pipeline_local --demo

# Live voice
python -m tests.test_pipeline_local --mic
python -m tests.test_pipeline_local --file sample_call.wav

# Search only, no LLM, no cost
python -m tests.test_pipeline_local --kb "amoxy sillin"
```

`--full-demo` output, abridged:

```
ACT 1 - Customer calls to order
Customer    : Hello, do you have Amoxil in stock?
MediTrack AI: Yes, we have Amoxil in stock. It's one thousand sixty two rupees
              fifty paise for a pack of twenty one capsules, but you'll need a
              valid prescription for it.
... name, phone and pickup/delivery collected one at a time ...
Customer    : Yes, please place the order
MediTrack AI: Your order has been placed! Your order number is one zero zero one.
Order record  : ORD-1001: 1 x Amoxil for Rathan Vengala - Rs.1,062.50 (awaiting_prescription)

ACT 2 - Pharmacist works the queue
  [NORMAL] ESC-20260809-0001 - prescription_required - Rathan Vengala
  Order ORD-1001 -> ready_for_pickup

ACT 3 - Customer calls back
Customer    : Hi, is my order 1001 ready?
MediTrack AI: Yes, your order one thousand one is ready for pickup at the counter.
```

## Testing results

```powershell
python -m tests.run_all          # everything, with a combined report
```

```
Suite                                             Tests  Passed      Time
------------------------------------------------------------------------------
Week 2 - knowledge base & query handling             25      25     1.15s
Week 3 - orders, inventory & escalation              34      34     2.23s
Week 4 - negative, security & recovery               42      42     9.40s
Week 4 - MVP validation benchmark                   131     131     1.89s
------------------------------------------------------------------------------
TOTAL                                               232     232    100.0%
```

`tests/test_negative.py` is the boundary/security/concurrency/recovery suite:
zero and negative quantities, an oversized utterance, a missing formulary, an
unauthenticated staff request, an unsigned Twilio webhook, a session left
idle past its TTL, silence on the line, two workers racing for the last pack,
and a corrupted `orders.json` — one test per defect found in the Week 3
review, plus guards that the happy path around each fix still works.

The Week 4 benchmark measures accuracy against labelled datasets rather than
just asserting pass/fail:

```powershell
python -m tests.test_validation
```

| Capability | Cases | Accuracy |
|---|---:|---:|
| Medicine search (mis-transcribed names) | 35 | 100.0% |
| Rejecting non-medicine phrases | 20 | 100.0% |
| Intent classification | 28 | 100.0% |
| Entity extraction (quantity, phone, name) | 21 | 100.0% |
| Speech-to-text noise filter | 13 | 100.0% |
| Clinical questions routed to pharmacist | 6 | 100.0% |
| Order workflow (end to end) | 8 | 100.0% |

No API key or network is needed for any of this — the LLM is not in the loop,
so the numbers are reproducible.

**What is not measured:** end-to-end speech accuracy (depends on the caller's
line and accent) and the LLM's phrasing quality (non-deterministic). Those were
validated by hand through the live microphone demos.

## Project structure

```
meditrack-ai/
├── app/
│   ├── main.py               FastAPI app, Twilio webhooks, REST API
│   ├── security.py           Staff API key check, Twilio signature verification
│   ├── storage.py            Cross-process lock, atomic write, corrupt-file recovery
│   ├── stt.py                Speech-to-Text + hallucination filtering
│   ├── knowledge_base.py     Formulary, fuzzy/phonetic search, availability
│   ├── query_handler.py      Intent, entity and customer-detail extraction
│   ├── order_flow.py         Order slot-filling state machine
│   ├── orders.py             Order records, inventory reservation, quantity validation
│   ├── escalation.py         Pharmacist escalation queue
│   ├── llm.py                Grounded response generation
│   ├── conversation.py       Per-call session store, TTL + count eviction
│   └── config.py             Environment-driven settings
├── data/
│   ├── medicines.json        28-medicine formulary (on-hand stock)
│   ├── pharmacy_info.json    Hours, delivery, services, policies
│   ├── orders.json           Order records (runtime, gitignored)
│   └── escalations.json      Pharmacist queue (runtime, gitignored)
├── tests/
│   ├── run_all.py               Combined test report
│   ├── test_knowledge_base.py   25 tests
│   ├── test_orders.py           34 tests
│   ├── test_negative.py         42 tests — boundary, security, concurrency, recovery
│   ├── test_validation.py       131-case accuracy benchmark
│   └── test_pipeline_local.py   Demo harness
├── ARCHITECTURE.md
├── requirements.txt
└── README.md
```

## API

`uvicorn app.main:app --reload --port 8000`, then **http://localhost:8000/docs**.

Every endpoint below except `/health` and `/voice/*` requires an `X-API-Key`
header matching `ADMIN_API_KEY`. `/voice/*` requires a valid Twilio signature
instead (see [Security](#security) below).

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /health` | none | Liveness, medicines loaded, orders, open escalations, active sessions, config warnings |
| `GET /kb/search?q=...` | API key | Fuzzy medicine search |
| `GET /kb/availability?name=...` | API key | Stock answer for a spoken name |
| `GET /inventory/check?name=...&quantity=N` | API key | On hand minus reserved |
| `GET /orders` | API key | Orders, filterable by `status` or `phone` |
| `POST /orders/{id}/status` | API key | Pharmacist transition |
| `GET /escalations` | API key | Pharmacist queue, priority ordered |
| `POST /escalations/{id}/status` | API key | Acknowledge or resolve |
| `POST /query/analyze` | API key | Intent, entities and the exact LLM grounding context |
| `POST /voice/incoming` | Twilio signature | First webhook on an incoming call |
| `POST /voice/process` | Twilio signature | Webhook after each `<Record>` |

## Security

- **Staff API.** Every non-`/health`, non-voice endpoint depends on
  `security.require_api_key()`, which compares `X-API-Key` against
  `ADMIN_API_KEY` with `hmac.compare_digest`. An unset key fails **closed** —
  a 503 explaining why, never a silently open endpoint. `DISABLE_AUTH=true`
  exists only for a fully local demo and must never be set anywhere reachable
  from a network.
- **Voice webhooks.** `security.verify_twilio_signature()` validates
  `X-Twilio-Signature` against `PUBLIC_BASE_URL` and the exact POST body using
  Twilio's own `RequestValidator`, so a request that did not genuinely come
  from Twilio is rejected with 403. `VALIDATE_TWILIO_SIGNATURE=false` is the
  equivalent local-only escape hatch.
- **Input limits.** `MAX_QUERY_CHARS` / `MAX_QUERY_TOKENS` bound how much text
  one turn can force the medicine search to scan; `MAX_ORDER_QUANTITY` bounds
  how many packs one phone order can request without a pharmacist.
- **Cross-process safety.** `storage.file_lock()` takes an OS-level advisory
  lock around every read-modify-write of `orders.json` / `escalations.json`,
  so running under `uvicorn --workers N` cannot oversell the same last pack.
- **Recovery.** Every write keeps a `.bak` snapshot; a damaged JSON file is
  recovered from it automatically, or quarantined and started clean if the
  backup is unusable too — the app never crashes at startup over a bad file.

## Business value

**The problem.** Pharmacy counter staff are interrupted constantly by calls
asking the same few things — is this in stock, what does it cost, is my order
ready. Each call pulls a pharmacist away from dispensing, and calls that arrive
while the line is busy are simply lost.

**What this addresses.**

| Value | How |
|---|---|
| Staff time returned to dispensing | Routine stock, price and status calls are answered without a human |
| No missed calls | The agent answers every call in parallel, including out of hours |
| Orders captured, not lost | An enquiry becomes a structured order record with customer details |
| Fewer stock errors | Reservations prevent promising the same pack twice |
| Safety preserved | Clinical questions are never answered by the AI; they queue for the pharmacist |
| Audit trail | Every order and escalation is tied to the call it came from |

**Where it does not apply.** Anything clinical, anything requiring professional
judgement, and dispensing itself. The design goal is to protect pharmacist time,
not to replace pharmacist decisions.

## Challenges encountered

**Speech recognition mangles drug names.** "Amoxicillin" came back as
"a master cylinder" from the free engine, and Whisper produced spellings like
`amoxy sillin`. Solved with a phonetic index plus literal-similarity guards —
measured at 100% recall over 35 mis-transcriptions, with zero false matches
across 20 ordinary call phrases.

**Whisper hallucinates on silence.** With no language pinned it transcribed
English speech into Tamil, Hindi and Chinese script, and emitted memorised
YouTube subtitle boilerplate during pauses — which the agent then tried to
answer. Fixed by pinning the language, priming with pharmacy vocabulary and
filtering known hallucination patterns.

**The noise filter then ate phone numbers.** It counted letters, and a dictated
phone number has none, so every number a caller gave was silently discarded and
the agent kept re-asking. A good reminder that a safety filter needs its own
negative tests — it now counts digits too, covered by regression tests.

**Language models will not hold a form in their head.** Early attempts to let
the model collect order details produced invented phone numbers and forgotten
quantities. Moving the sequence, arithmetic and stock checks into a
deterministic state machine — leaving the model only the wording — made the
behaviour testable and stopped the invention.

**A green test suite only proves the paths it exercises.** A structured review
ran 190 cases by hand against the Week 3 build and found 18 failures — a
negative quantity persisted with a negative total, an unauthenticated API
serving the full order book, a missing formulary crashing the process at
import, "daily" matching Centrum through its "daily vitamin" alias — while the
offline suite stayed green throughout, because it only ever asked the happy
path. Every one is now fixed and pinned as a named test in
`tests/test_negative.py`, and the full detail is in
[ARCHITECTURE.md §9](ARCHITECTURE.md#9-week-4-hardening).

**Small ambiguities break flows in non-obvious ways.** "Yes, please place *the
order*" matched the order-status pattern and silently swallowed the
confirmation. "It's Rattan" parsed as "S Rattan". "Ten *inhalers*" read as one
pack because that unit was missing from the list — a silent under-order, the
worst failure mode for an ordering system. Each was found by running the demos
rather than by reading the code.

## Future improvements

**Next (Stage III)**
- Prescription image upload and verification, closing the `awaiting_prescription` loop
- SMS confirmation and status notifications back to the customer
- Move orders and sessions from JSON to a database with real transactions —
  `storage.file_lock()` safely serialises multiple workers on one host, but
  not multiple hosts sharing the files over a network filesystem

**Beyond**
- Streaming speech so the caller can interrupt, instead of fixed-window recording
- Read-back confirmation of phone numbers and order numbers
- Integration with the pharmacy's real POS and stock system in place of the JSON formulary
- Multilingual support — the pipeline pins English today, and Indian pharmacy
  callers often switch language mid-sentence
- A pharmacist web dashboard over the existing escalation and order endpoints
- Repeat-customer recognition by phone number, for faster refill ordering

## Full phone call demo (Twilio)

1. `uvicorn app.main:app --reload --port 8000`
2. `ngrok http 8000`
3. Put the ngrok HTTPS URL into `.env` as `PUBLIC_BASE_URL`.
4. In the [Twilio Console](https://console.twilio.com), set your number's
   **"A call comes in"** to `https://<your-ngrok-url>/voice/incoming` (POST).
5. Call and place an order. Orders carry the Twilio call SID, so any record can
   be traced back to the recording.

## Resetting demo data

Order and escalation files are written at runtime and gitignored:

```powershell
Remove-Item data\orders.json, data\escalations.json -ErrorAction SilentlyContinue
```
