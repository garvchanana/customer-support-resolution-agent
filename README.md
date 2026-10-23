# Customer Support Resolution Agent

A backend AI engineering project that classifies customer-support messages, retrieves relevant policies, investigates order information, assesses refund eligibility, and generates structured support resolutions.

Built with Python, FastAPI, LangGraph, PostgreSQL, SQLAlchemy, and machine-learning classification models. The project emphasizes deterministic business rules, policy-grounded responses, regression testing, and safe evaluation of LLM-generated drafts.

**Important:** This application assesses refund eligibility; it does not issue refunds. Refunds requiring approval must be reviewed by a human.

## Features

- **FastAPI backend:** API endpoints for support resolution, message classification, ticket operations, and refund assessment.
- **LangGraph workflow:** Structured processing for classification, clarification, policy retrieval, order investigation, and response formulation.
- **PostgreSQL persistence:** Demo customers, orders, returns, payments, and tickets, plus separate historical dataset tables.
- **ML classification suggestions:** Intent and category predictions support the workflow but do not independently authorize business actions.
- **Policy retrieval:** Keyword-based retrieval by default, with an optional semantic retrieval implementation.
- **Refund assessment:** Evaluates available order, payment, return, and policy information.
- **LLM shadow mode:** Optional Groq-powered response drafts are evaluated without replacing the authoritative deterministic response.
- **Safety checks:** Rejects certain unsupported claims and commitments in generated drafts.
- **Automated tests and evaluations:** Regression tests, retrieval benchmarks, routing-safety checks, and LLM shadow evaluations.

## Technology stack

- Python
- FastAPI and Uvicorn
- LangGraph
- PostgreSQL
- SQLAlchemy and Alembic
- Pydantic
- scikit-learn
- sentence-transformers
- Groq Python SDK
- pytest and HTTPX

## Architecture

The support workflow follows a controlled sequence:

1. Classify the incoming customer message.
2. Generate advisory ML classification suggestions where configured.
3. Determine whether clarification or a supported workflow is required.
4. Retrieve relevant policy evidence.
5. Investigate available order and refund information.
6. Assess the case using deterministic business rules.
7. Formulate the authoritative response.
8. Optionally generate an LLM draft in shadow mode.

The deterministic workflow remains authoritative. An LLM-generated draft does not change the final response or outcome.

### Resolution outcomes

- `resolved` — the workflow has formulated a response for the supported request.
- `clarification_required` — additional information is needed.
- `human_review_required` — a human must review the case before the relevant action proceeds.
- `unsupported` — the request is outside the supported workflow.

The exact outcome depends on the workflow and the available verified information.

## Project structure

```text
customer-support-resolution-agent/
├── app/
│   ├── agent/
│   │   ├── classification.py
│   │   ├── response_generation.py
│   │   ├── retrieval.py
│   │   ├── semantic_retrieval.py
│   │   └── workflow.py
│   ├── services/
│   │   └── refunds.py
│   ├── data.py
│   ├── db.py
│   ├── main.py
│   ├── models.py
│   ├── olist_models.py
│   └── schemas.py
├── artifacts/
│   ├── category_baseline/
│   ├── grouped_baselines/
│   └── intent_baseline/
├── datasets/
│   ├── support_eval.json
│   └── support_challenge.json
├── knowledge_base/
│   ├── damaged_item_policy.md
│   └── refund_policy.md
├── migrations/
│   └── versions/
├── scripts/
├── tests/
├── .env.example
├── .gitignore
├── alembic.ini
├── requirements.txt
└── README.md
```

Additional scripts support dataset inspection and preparation, database checks, model training, routing evaluation, retrieval evaluation, and LLM shadow evaluation.

## Requirements

- Python compatible with the installed dependencies
- pip
- PostgreSQL
- Git

The commands below use **PowerShell on Windows**. Docker and WSL are not required for this setup.

## Setup

### 1. Clone the repository

```powershell
git clone <repository-url>
cd customer-support-resolution-agent
```

Replace `<repository-url>` with the URL of your repository.

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell prevents activation because of its execution policy, follow your organization's or system administrator's approved guidance for running virtual-environment activation scripts.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example environment file:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and configure the database connection string for your local PostgreSQL installation.

The project uses the following configuration options:

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `SUPPORT_RETRIEVAL_MODE` | Retrieval implementation: `keyword` or `semantic` |
| `SUPPORT_LLM_ENABLED` | Enables or disables LLM shadow-mode draft generation |
| `GROQ_API_KEY` | Optional Groq API key for live draft generation |
| `GROQ_MODEL` | Groq model identifier |

The example configuration uses keyword retrieval and disables LLM shadow mode by default. The Groq model is configured through `GROQ_MODEL`.

Keep `.env` and all credentials out of version control. Never put API keys directly into source code or paste them into public issue reports.

### 5. Prepare PostgreSQL

Create a PostgreSQL database and an application database user with the required permissions.

Set `DATABASE_URL` in `.env` to the connection string for that database. Use a URL-encoded password if it contains characters that require encoding.

Apply the database migrations:

```powershell
python -m alembic upgrade head
```

Seed the demo records:

```powershell
python -m scripts.seed_data
```

Run the seed script when the demo database needs its fixture records. Review the script before rerunning it against any database containing data you need to preserve.

### 6. Model artifacts and datasets

The repository includes trained classification model artifacts and evaluation JSON files.

The local raw and processed dataset directories are excluded from Git. Dataset preparation scripts are provided under `scripts/` for workflows that require those source files.

Some scripts require the relevant source datasets to be obtained separately. Follow each script's expected inputs and options rather than assuming all datasets are included in the repository.

The application also includes separate historical dataset tables for analysis. Historical e-commerce records must not be treated as proof that an individual demo order qualifies for a refund.

## Run the API

Start the development server:

```powershell
uvicorn app.main:app --reload
```

Open the interactive API documentation:

`http://127.0.0.1:8000/docs`

Use the documented request schemas to explore the support-resolution, classification, ticket, and refund-assessment endpoints.

The demo uses the `X-Demo-Customer-ID` request header for customer context where applicable. This is a demonstration mechanism, not production authentication.

Stop the server with `Ctrl+C`.

## Run the demo

From the project root:

```powershell
python run_agent_demo.py
```

This runs the project's command-line demonstration. Use the API documentation for interactive endpoint testing.

## Run tests and evaluations

Run the full regression suite:

```powershell
python -m pytest -q
```

Run the agent evaluation:

```powershell
python -m scripts.evaluate_agent
```

Run the challenge evaluation:

```powershell
python -m scripts.evaluate_challenge
```

Run the routing-safety evaluation:

```powershell
python -m scripts.evaluate_routing_safety
```

Run the retrieval benchmark:

```powershell
python -m scripts.evaluate_retrieval
```

Run the live LLM shadow evaluation:

```powershell
python -m scripts.evaluate_llm_shadow
```

The LLM shadow evaluation requires a valid Groq API key and can consume API quota. It is separate from the deterministic regression tests.

## Retrieval evaluation

The retrieval benchmark contains seven test cases covering refund policy, damaged-item policy, mixed requests, and unsupported queries.

Latest recorded benchmark results:

| Metric | Keyword | Semantic |
|---|---:|---:|
| Expected-source hit rate | 100.0% | 100.0% |
| Unsupported-query false-positive rate | 0.0% | 0.0% |
| Irrelevant-result rate on supported queries | 23.1% | 28.6% |

Keyword retrieval remains the default because it produced fewer irrelevant results in this small benchmark.

These figures describe only the tested cases. A seven-case benchmark is too small to establish general retrieval performance or production reliability. Results should be reevaluated when the policy corpus or retrieval implementation changes.

## LLM shadow-mode evaluation

The project optionally uses Groq to generate candidate customer-support response drafts.

In shadow mode:

- The deterministic response and outcome remain authoritative.
- The LLM receives verified facts and retrieved policy evidence.
- Drafts are subject to validation and a safety gate.
- Drafts that fail validation or safety checks are discarded.
- The LLM does not issue refunds, approve refunds, or change the deterministic outcome.

### Latest recorded evaluation

Three scenarios are used by the current shadow evaluation:

| Scenario | Observed behavior |
|---|---|
| Refund request | A draft was generated and the deterministic response and outcome remained unchanged. |
| Damaged-item refund request | The draft was rejected by the safety gate; the deterministic response and outcome remained unchanged. |
| Unsupported hotel-booking request | No draft was returned; the deterministic response and outcome remained unchanged. |

The damaged-item rejection is not proof that the safety gate catches every unsafe claim. It demonstrates that a rejection path is exercised for this scenario.

The current safety checks are not a guarantee of factual correctness. Broader tests are needed for incorrect currency interpretation, unsupported recording claims, invented policy instructions, refund eligibility misstatements, and unsupported follow-up promises.

Keep `SUPPORT_LLM_ENABLED=false` by default until broader evaluation supports any change in operating mode.

## Data and evaluation limitations

- The Olist dataset represents historical Brazilian e-commerce activity. It does not establish the refund eligibility of a particular order in this application's demo workflow.
- The Bitext customer-support dataset is synthetic or curated and is not verified real customer conversation data.
- Classification metrics on these datasets do not establish performance on representative production support traffic.
- The retrieval benchmark has only seven cases and should be treated as an initial regression baseline.
- The LLM shadow evaluation covers only a small number of scenarios.
- Automated tests demonstrate behavior for tested cases; they do not prove production readiness.

## Security and operational boundaries

- Keep `.env`, credentials, and API keys out of version control.
- The `X-Demo-Customer-ID` header is not a substitute for authenticated identity and server-side authorization.
- Validate customer ownership and authorization before exposing order information or permitting actions.
- Refund eligibility assessment does not execute payments or refunds.
- Human approval is required where the workflow returns `human_review_required`.
- Do not treat generated text as a source of truth for policy, order data, eligibility, or authorization.
- Before production deployment, add appropriate authentication, authorization, monitoring, error handling, auditability, and broader security and reliability evaluation.

## Current validation

Latest recorded local regression suite:

- **66 tests passed.**

Latest recorded retrieval benchmark:

- Keyword expected-source hit rate: **100.0%**
- Keyword unsupported-query false-positive rate: **0.0%**
- Keyword irrelevant-result rate on supported queries: **23.1%**

The LLM shadow evaluation preserves deterministic responses and outcomes in the three recorded scenarios. The damaged-item draft was rejected, and the unsupported request produced no draft.

Rerun the tests and evaluations after code or policy changes and update this section with the new results. These results are an initial portfolio baseline, not a production-readiness certification.
