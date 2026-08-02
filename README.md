# MediTrack AI — Week 2

**AI Voice Agent for Pharmacy Enquiry & Order Automation**
Stage II, Week 2: Pharmacy Knowledge Base & Medicine Query Handling

## What's implemented this week

Per the Week 2 plan, the agent now answers pharmacy-specific questions from a
real formulary instead of improvising:

```
Customer Query → Intent + Entity Extraction → Knowledge Base Search → AI Response
```

- **Pharmacy knowledge base** — `data/medicines.json` (28 medicines) and
  `data/pharmacy_info.json` (hours, delivery, services, policies), loaded and
  indexed by `app/knowledge_base.py`.
- **Medicine information** — brand and generic names, aliases, category, form,
  strength, pack size, price, stock level, prescription status, common uses,
  storage instructions, shelf location, and substitutes.
- **Medicine search** — fuzzy, phone-tolerant matching. A phonetic index folds
  the spelling variants speech-to-text produces, so `amoxy sillin`,
  `paracetemol` and `ventalin` all resolve correctly. Falls back to searching
  by symptom/category when no name matches ("something for a headache").
- **Availability queries** — `check_availability()` returns in stock / low
  stock / out of stock, and offers in-stock substitutes when an item is out.
- **Improved response accuracy** — every LLM turn is grounded on a knowledge
  base context block. The system prompt forbids inventing medicines, prices
  or stock, and clinical questions (dosage, interactions) are routed to the
  pharmacist. If no API key is configured, replies are composed from the
  knowledge base directly, so the pipeline never dead-ends.

Week 1's pipeline (Twilio voice → Whisper STT → LLM → spoken reply) is
unchanged and still works; Week 2 slots the knowledge base into the middle of
it. Order creation and payment remain out of scope until Week 3.

## Project structure

```
meditrack-ai/
├── app/
│   ├── main.py               FastAPI app, Twilio webhooks, knowledge base endpoints
│   ├── stt.py                Speech-to-Text (Whisper + fallback)
│   ├── knowledge_base.py     Formulary loading, fuzzy/phonetic search, availability
│   ├── query_handler.py      Intent detection, entity extraction, LLM grounding context
│   ├── llm.py                Grounded response generation (OpenAI or Anthropic)
│   ├── conversation.py       In-memory per-call session store
│   └── config.py             Environment-driven settings
├── data/
│   ├── medicines.json        Medicine formulary + stock snapshot
│   └── pharmacy_info.json    Store hours, delivery, services, policies
├── tests/
│   ├── test_pipeline_local.py   Demo the pipeline without Twilio
│   └── test_knowledge_base.py   21 tests for search, intents, grounding
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # then fill in your API keys
```

Set `LLM_PROVIDER` to `openai` or `anthropic` and supply the matching key.
Whisper STT needs `OPENAI_API_KEY`; without it, STT falls back to the free
SpeechRecognition/Google engine. **Without any LLM key the demo still runs** —
replies are generated from the knowledge base alone.

## Week 2 demonstration

The scripted demo walks six customer queries through all three stages:

```bash
python -m tests.test_pipeline_local --demo          # with the grounding context shown
python -m tests.test_pipeline_local --demo --quiet  # just query → search → response
```

Sample output:

```
Customer Query   : I need two boxes of augmentin six twenty five
  [1] Intent     : availability
  [2] KB Search  : Augmentin (heard 'augmentin', score 1.0)
      Stock      : Augmentin is currently out of stock. Available alternatives: Amoxil, Zithromax.
      Quantity   : 2
  [3] AI Response: I'm sorry, but Augmentin is currently out of stock. However, we do
                   have Amoxil and Zithromax available as alternatives. Would you like
                   more information on either of those?
```

Other local modes:

```bash
# Search the knowledge base only, no LLM call
python -m tests.test_pipeline_local --kb "amoxy sillin"

# One custom query through the full pipeline
python -m tests.test_pipeline_local --text "Do you have anything for a cough?"

# Week 1 modes still work
python -m tests.test_pipeline_local --mic
python -m tests.test_pipeline_local --file sample_call.wav
```

## Tests

```bash
python -m tests.test_knowledge_base                  # no extra install needed
pip install pytest && python -m pytest tests -q      # optional, if you prefer pytest
```

Covers exact/alias/phonetic matching, misheard-name recall, rejection of
nonsense and of ordinary English words, the three availability states,
substitute suggestions, intent detection, quantity/strength/form extraction,
clinical routing, and that the grounding context never invents a medicine.

## Knowledge base endpoints

Start the server with `uvicorn app.main:app --reload --port 8000`, then:

```bash
curl "http://localhost:8000/health"
curl "http://localhost:8000/kb/search?q=amoxy%20sillin"
curl "http://localhost:8000/kb/availability?name=augmentin"
curl "http://localhost:8000/kb/medicines?category=Antibiotic"
curl -X POST "http://localhost:8000/query/analyze" -d "text=do you have two boxes of ventolin"
```

`/query/analyze` shows the intent, extracted entities and the exact grounding
context the LLM receives — useful for demonstrating *why* a reply says what it
says.

## Full phone call demo (with Twilio)

1. `uvicorn app.main:app --reload --port 8000`
2. `ngrok http 8000`
3. Put the ngrok HTTPS URL into `.env` as `PUBLIC_BASE_URL`.
4. In the [Twilio Console](https://console.twilio.com), set your number's
   **"A call comes in"** to `https://<your-ngrok-url>/voice/incoming` (POST).
5. Call the number and ask about a medicine.

## Tuning the search

`KB_MATCH_THRESHOLD` in `.env` (default `0.62`) controls how confident a fuzzy
match must be before a spoken word is accepted as a medicine name. Lower it if
your STT mangles names badly; raise it if the agent guesses wrong medicines.

## Next steps (Week 3)

- Order creation and reservation against the stock levels
- Prescription upload and verification flow
- Persisting sessions and orders beyond process memory
