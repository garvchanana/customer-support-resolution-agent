# Customer Support Resolution Agent

A policy-aware customer support API built with FastAPI, LangGraph, PostgreSQL, and machine-learning-assisted ticket classification. Deployed on Render with Neon PostgreSQL and optional Groq-powered response drafting.

**Live API:** https://customer-support-resolution-agent-qrjc.onrender.com  
**API documentation:** https://customer-support-resolution-agent-qrjc.onrender.com/docs  
**Health check:** https://customer-support-resolution-agent-qrjc.onrender.com/health

## Overview

The Customer Support Resolution Agent processes customer support requests through a policy-aware workflow that combines order data, policy retrieval, deterministic response generation, and ML-assisted routing suggestions.

The system is designed to keep sensitive decisions under human control. Refund eligibility assessment is separate from refund execution, and ML predictions do not independently authorize refunds.

An optional Groq integration generates response drafts in **shadow mode**. Drafts are checked for unsupported claims and unsafe commitments, while the deterministic response remains authoritative.

### Key capabilities

- Support request classification and category suggestions.
- Ticket creation and retrieval.
- Order-based refund eligibility assessment.
- Policy-grounded response generation.
- Human-review requirements for refund decisions.
- ML intent and category suggestions with confidence and margin metrics.
- Keyword-based policy retrieval by default, with optional semantic retrieval.
- Optional Groq-powered response drafting in shadow mode.
- Validation of generated drafts against unsupported-claim safety checks.
- PostgreSQL persistence with versioned Alembic migrations.
- Interactive API documentation through FastAPI Swagger UI.
- Automated tests covering API behavior, refund safety, retrieval, classification, and shadow-mode execution.

## Technology stack

| Layer | Technology |
|---|---|
| API | FastAPI |
| API server | Uvicorn |
| Agent workflow | LangGraph |
| Database | Neon PostgreSQL |
| ORM | SQLAlchemy |
| Schema migrations | Alembic |
| Validation | Pydantic |
| ML utilities | scikit-learn |
| Optional semantic retrieval | sentence-transformers |
| Optional LLM provider | Groq |
| Hosting | Render |
| Source control | Git and GitHub |
| Testing | pytest |

## Architecture

```text
Client / API consumer
        |
        v
Render Web Service
        |
        v
FastAPI application
        |
        +---- Ticket endpoints
        |
        +---- Refund assessment
        |
        +---- Support resolution workflow
                   |
                   +---- Policy retrieval
                   |
                   +---- ML routing suggestions
                   |
                   +---- Deterministic response
                   |
                   +---- Optional Groq shadow draft
                   |
                   +---- Draft safety validation
                   |
                   +---- Human-review controls
        |
        v
SQLAlchemy
        |
        v
Neon PostgreSQL
```

The default production retrieval mode is `keyword`. Semantic retrieval is optional and requires the dependencies in `requirements-semantic.txt`.

## Live deployment

The application is deployed as a Render Web Service connected to a Neon PostgreSQL database.

- **Base URL:** https://customer-support-resolution-agent-qrjc.onrender.com
- **Health endpoint:** `/health`
- **Swagger documentation:** `/docs`
- **OpenAPI schema:** `/openapi.json`

Check the health endpoint:

```powershell
curl.exe -i https://customer-support-resolution-agent-qrjc.onrender.com/health
```

Expected response:

```json
{
  "status": "ok"
}
```

The health check verifies application availability. It does not, by itself, prove database connectivity or confirm that every business workflow is functioning.

## API endpoints

Refer to the live `/docs` page for the authoritative request schemas, HTTP methods, headers, and response models.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Application health check |
| `POST /support/classify` | Classify a support request |
| `POST /tickets` | Create a support ticket |
| `GET /tickets/{ticket_id}` | Retrieve a ticket |
| `GET /orders/{order_id}/refund-assessment` | Assess refund eligibility |
| `POST /support/resolve` | Run the support-resolution workflow |

Some endpoints require specific request headers or body fields. Use the Swagger documentation to confirm their current contracts.

## Refund safety and human approval

Refund assessment is separate from refund execution.

For example:

```json
{
  "eligible": true,
  "amount_minor": 499900,
  "currency": "INR",
  "reason": "ELIGIBLE_PENDING_HUMAN_APPROVAL",
  "requires_approval": true
}
```

This means the order meets the configured demo eligibility checks but still requires human approval.

`amount_minor` represents the amount in the currency's minor units. For INR, `499900` corresponds to ₹4,999.00.

The system also checks order status. For example, an order that is still shipped can receive an `ORDER_NOT_DELIVERED` assessment and remain ineligible for a refund under the configured demo policy.

ML intent and category predictions are suggestions only. They do not independently authorize refunds or override the refund assessment.

The deployed demo verifies support-resolution and refund-assessment behavior. It does not represent a live payment-provider integration and does not execute actual refunds.

## Groq integration and shadow mode

The application supports optional Groq-powered response drafting through the `SUPPORT_LLM_ENABLED` configuration.

When enabled, the shadow node:

1. Receives the customer message, verified workflow facts, and retrieved policy evidence.
2. Requests a draft from the configured Groq model.
3. Parses and validates the returned JSON response.
4. Checks the draft for unsupported commitments, unverified recording claims, unsupported procedural requests, and internal currency-unit wording.
5. Returns an accepted draft or treats the draft as unavailable.
6. Leaves the existing deterministic response unchanged.

The deterministic response remains the authoritative response returned by the workflow. An accepted draft is **not** equivalent to a refund approval, a completed support action, or permission to replace the deterministic response.

### Runtime diagnostics

The shadow node emits diagnostic logs to help verify production execution:

| Log marker | Purpose |
|---|---|
| `SHADOW_NODE_EXECUTED` | Confirms the shadow node ran |
| `SHADOW_NODE_CONFIG` | Reports whether shadow mode is enabled |
| `SHADOW_NODE_INPUT` | Reports message presence and policy-evidence count |
| `Groq shadow draft status=accepted` | Indicates the draft passed the validation path |
| `Groq shadow draft status=unavailable` | Indicates no usable draft was returned |
| `Groq shadow draft status=error` | Indicates an exception was caught by the workflow node |

Provider-specific warnings may also identify missing configuration, invalid output, safety rejection, or a provider failure. Logs should not include API keys or full customer messages.

### Configuration

Example local settings:

```env
SUPPORT_LLM_ENABLED=false
GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-120b
```

Keep the feature disabled unless you intentionally want to enable LLM drafting. When enabled, configure the API key through a secure environment variable and verify the behavior through tests and runtime logs.

## Local setup — Windows PowerShell

### 1. Clone the repository

```powershell
git clone https://github.com/garvchanana/customer-support-resolution-agent.git
cd customer-support-resolution-agent
```

### 2. Create and activate a virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use an appropriate execution-policy setting for your environment or invoke the virtual environment's Python directly.

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For optional semantic retrieval:

```powershell
pip install -r requirements-semantic.txt
```

### 4. Configure environment variables

Create a local `.env` file from the example:

```powershell
Copy-Item .env.example .env
```

Configure `DATABASE_URL` with your Neon PostgreSQL connection string or a suitable local PostgreSQL connection string.

A SQLAlchemy URL compatible with Psycopg 3 can use this format:

```env
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST/DATABASE?sslmode=require
SUPPORT_RETRIEVAL_MODE=keyword
SUPPORT_LLM_ENABLED=false
GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-120b
```

Replace the placeholders locally. Keep `.env` private and never commit credentials to source control.

### 5. Apply database migrations

```powershell
python -m alembic upgrade head
```

Check the current migration revision:

```powershell
python -m alembic current
```

### 6. Load synthetic demo data

```powershell
python -m scripts.seed_data
```

Review the seed script before using it outside a demo environment.

### 7. Start the API locally

```powershell
uvicorn app.main:app --reload
```

Open the local endpoints:

- API: http://127.0.0.1:8000
- Swagger UI: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

## Environment variables

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `SUPPORT_RETRIEVAL_MODE` | Retrieval mode; `keyword` is the current deployment default |
| `SUPPORT_LLM_ENABLED` | Enables or disables optional LLM drafting |
| `GROQ_API_KEY` | Groq API credential when LLM drafting is enabled |
| `GROQ_MODEL` | Configured Groq model name |
| `PYTHON_VERSION` | Python version selected for the Render build |

Keep secret values in local `.env` or the Render service's environment settings. `.env.example` should contain placeholders only.

## Render deployment configuration

The production service uses the following configuration.

**Build command**

```bash
pip install -r requirements.txt
```

**Start command**

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Configure the required environment variables in the Render dashboard. `DATABASE_URL` must point to the intended Neon database.

The application listens on the port supplied by Render through `$PORT`. After deployment, verify `/health`, `/docs`, and a database-backed endpoint. Check the Render runtime logs if the service fails to start or a workflow behaves unexpectedly.

## Database migrations and seed data

Alembic migration files live under `migrations/versions/`.

Apply schema changes with:

```powershell
python -m alembic upgrade head
```

Load synthetic demo records with:

```powershell
python -m scripts.seed_data
```

Do not run migrations or seed scripts automatically on every application startup unless that behavior is deliberately designed and tested.

## Testing

Run the full test suite from the project root:

```powershell
python -m pytest -q
```

Run the shadow-mode tests separately:

```powershell
python -m pytest tests/test_llm_shadow_mode.py -q
```

### Verified test result

The latest full local test run completed successfully:

```text
80 passed in 22.69s
```

The shadow-mode tests also passed independently.

The test suite covers API behavior, refund safety, classification, retrieval, response generation, and shadow-mode execution. Tests should use synthetic data and an isolated test database where integration testing requires a database.

## Repository structure

```text
customer-support-resolution-agent/
|-- app/
|   |-- main.py
|   |-- db.py
|   `-- agent/
|-- migrations/
|   |-- env.py
|   `-- versions/
|-- scripts/
|   `-- seed_data.py
|-- tests/
|-- requirements.txt
|-- requirements-semantic.txt
|-- Procfile
|-- alembic.ini
|-- .env.example
|-- .gitignore
`-- README.md
```

This is a high-level view of the principal files; the exact contents of subdirectories may evolve as the project develops.

## Security and operational notes

- Never commit `.env`, database passwords, or API keys.
- Use synthetic data in demos and tests.
- Keep refund approval separate from eligibility assessment.
- Treat ML predictions as suggestions, not authorization.
- Restrict and authenticate sensitive endpoints before production use.
- Add appropriate rate limits, monitoring, and error handling before handling real customer traffic.
- Avoid logging customer messages, credentials, or sensitive provider payloads.
- Review database access permissions and backup requirements.
- Review Render and Neon service limits and current pricing before relying on a free-tier deployment for critical workloads.
- Do not treat a successful health check as a complete database or business-workflow test.

## Deployment verification

The following checks have succeeded during the latest verification:

- Render service returned to a live state after deployment.
- Health endpoint responded successfully.
- Support-resolution request returned HTTP `200 OK`.
- Refund assessment returned a result consistent with the order status and configured demo policy.
- Shadow node executed with `SUPPORT_LLM_ENABLED` enabled.
- Runtime diagnostics confirmed a customer message was present and three policy-evidence items were supplied for the verified request.
- Groq generated a draft that was accepted by the validation path.
- The deterministic response remained authoritative.
- The full local test suite passed: **80 passed**.

These checks validate the demonstrated application workflow, not production readiness for processing real customer refunds or handling sensitive customer traffic at scale.