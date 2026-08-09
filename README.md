# MediTrack AI — Week 3

**AI Voice Agent for Pharmacy Enquiry & Order Automation**
Stage II, Week 3: Order Creation & Pharmacy Workflow Integration

## What's implemented this week

The agent now converts an enquiry into a real, persisted order:

```
Request → Availability Check → Order Creation → Confirmation
```

- **Collect customer information** — name, phone number, pickup vs delivery and
  delivery address are gathered across turns by a slot-filling state machine
  (`app/order_flow.py`), which asks for exactly one missing detail at a time.
- **Capture medicine details and quantity** — the Week 2 search resolves the
  medicine; quantity extraction covers every dispensing unit (packs, boxes,
  strips, inhalers, pens, bottles) in both digits and words ("ten inhalers",
  "half a dozen strips"). A medicine named on an earlier turn carries forward,
  so "I'd like to order two packs" works without repeating the name.
- **Inventory checking** — `OrderStore.check_stock()` answers *on hand minus
  everything already reserved*, so two callers cannot both take the last pack.
  Orders are re-checked at creation time.
- **Order records** — persisted to `data/orders.json` with items, line totals,
  delivery fee, status and the call they came from. Order numbers are short
  (`ORD-1001`) so they can be read back over the phone.
- **Pharmacist escalation** — `app/escalation.py` queues anything the AI must
  not decide alone, prioritised for the pharmacist on duty.

Weeks 1 and 2 are unchanged; the order workflow sits on top of them.

### Escalation triggers

| Trigger | Priority | What the caller hears |
|---|---|---|
| Dosage, interaction or diagnosis question | High | Pharmacist will call back |
| Caller asks for a human | High | Being passed to the pharmacist |
| Refrigerated (cold chain) item ordered | High | Pharmacist handles it |
| Prescription-only medicine ordered | Normal | Order held until prescription received |
| Fewer packs on the shelf than requested | Normal | What is available, plus alternatives |
| Medicine out of stock | Low | Alternatives, pharmacist will follow up |

### Order statuses

`draft` → `confirmed` → `ready_for_pickup` / `out_for_delivery` → `completed`,
with `awaiting_prescription`, `escalated` and `cancelled` as branches. Every
status except `completed` and `cancelled` holds stock in reserve.

## Project structure

```
meditrack-ai/
├── app/
│   ├── main.py               FastAPI app, Twilio webhooks, KB + order endpoints
│   ├── stt.py                Speech-to-Text (Whisper + noise filtering)
│   ├── knowledge_base.py     Formulary loading, fuzzy search, availability
│   ├── query_handler.py      Intent detection, entity + customer detail extraction
│   ├── order_flow.py         Order slot-filling state machine
│   ├── orders.py             Order records, inventory reservation, persistence
│   ├── escalation.py         Pharmacist escalation queue
│   ├── llm.py                Grounded response generation
│   ├── conversation.py       Per-call session store
│   └── config.py             Environment-driven settings
├── data/
│   ├── medicines.json        28-medicine formulary (on-hand stock)
│   ├── pharmacy_info.json    Hours, delivery, services, policies
│   ├── orders.json           Order records (created at runtime, gitignored)
│   └── escalations.json      Pharmacist queue (created at runtime, gitignored)
├── tests/
│   ├── test_pipeline_local.py   Demo harness (no Twilio needed)
│   ├── test_knowledge_base.py   24 tests — search, intents, grounding
│   └── test_orders.py           27 tests — orders, inventory, escalation
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate           # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # then fill in your API keys
```

Set `LLM_PROVIDER` to `openai` or `anthropic` and supply the matching key.
**Without any LLM key the demos still run** — replies are composed from the
knowledge base and order state alone.

## Week 3 demonstration

```bash
python -m tests.test_pipeline_local --order-demo        # request → availability → order → confirmation
python -m tests.test_pipeline_local --escalation-demo   # the four pharmacist hand-off paths
```

Sample output:

```
Stock before call : 1 pack of Panadol can be reserved, 148 available.

Customer   : Hi, do you have Panadol in stock?
MediTrack AI: Yes, we have Panadol in stock! Two hundred ninety seven rupees
              fifty paise for a pack of twenty tablets.

Customer   : Great, I would like to order two packs please
  [order]  : items=2xPanadol
  [awaiting]: name
MediTrack AI: Sure! Can I have the name to put the order under?

... name, phone and pickup/delivery collected one at a time ...

Customer   : Yes, please place the order
MediTrack AI: Your order has been placed! Your order number is one zero zero one.
              The total is five hundred ninety five rupees.

Stock after call  : 146 available.
Order record      : ORD-1001: 2 x Panadol for Rathan Vengala - Rs.595.00 (confirmed)
```

Earlier weeks' demos still work:

```bash
python -m tests.test_pipeline_local --demo              # Week 2 knowledge base demo
python -m tests.test_pipeline_local --kb "amoxy sillin" # search only, no LLM
python -m tests.test_pipeline_local --mic               # live microphone
python -m tests.test_pipeline_local --file sample_call.wav
```

## Tests

```bash
python -m tests.test_knowledge_base     # 24 tests
python -m tests.test_orders             # 27 tests
```

`test_orders.py` uses temporary files throughout, so running it never touches
your real order or escalation data.

## Order and escalation endpoints

Start the server with `uvicorn app.main:app --reload --port 8000`, then open
**http://localhost:8000/docs**.

| Endpoint | Purpose |
|---|---|
| `GET /inventory/check?name=ventolin&quantity=10` | Can we supply it? (on hand minus reserved) |
| `GET /orders` | All orders, filterable by `status` or `phone` |
| `GET /orders/{id}` | One order record |
| `POST /orders/{id}/status` | Pharmacist transition, e.g. → `ready_for_pickup` |
| `GET /escalations` | Pharmacist queue, priority ordered |
| `POST /escalations/{id}/status` | Acknowledge or resolve an escalation |
| `GET /kb/search?q=...` | Week 2 fuzzy medicine search |
| `POST /query/analyze` | Intent, entities and the exact LLM grounding context |

## Design notes

**Stock is never mutated.** `data/medicines.json` holds what is physically on
the shelf and is treated as read-only. Reservations live in `data/orders.json`,
so availability is always `on_hand − reserved`. Cancelling an order releases its
stock automatically, and the formulary stays the single source of truth. Actual
deduction would happen at dispense time, through the pharmacy's POS.

**The state machine decides, the LLM speaks.** `order_flow.py` computes which
detail is missing and passes an explicit instruction to the model. The wording
stays natural, but the sequence, the arithmetic and the stock decisions are
deterministic and testable — the model is never asked to remember what it has
already collected.

## Full phone call demo (with Twilio)

1. `uvicorn app.main:app --reload --port 8000`
2. `ngrok http 8000`
3. Put the ngrok HTTPS URL into `.env` as `PUBLIC_BASE_URL`.
4. In the [Twilio Console](https://console.twilio.com), set your number's
   **"A call comes in"** to `https://<your-ngrok-url>/voice/incoming` (POST).
5. Call and place an order. Orders are tagged with the Twilio call SID, so the
   pharmacist can trace any record back to the recording.

## Next steps (Week 4)

- Prescription image upload and verification
- Order status notifications back to the customer (SMS)
- Moving orders and sessions from JSON files to a database
