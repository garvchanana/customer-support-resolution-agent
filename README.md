# Customer Support Resolution Agent

A policy-aware customer support API built with FastAPI, LangGraph, PostgreSQL, and machine-learning-assisted ticket classification. Deployed on Render with Neon PostgreSQL.

**Live API:** https://customer-support-resolution-agent-qrjc.onrender.com  
**API documentation:** https://customer-support-resolution-agent-qrjc.onrender.com/docs  
**Health check:** https://customer-support-resolution-agent-qrjc.onrender.com/health

## Overview

The Customer Support Resolution Agent helps process customer support requests using policy-aware workflows, order data, ticket management, and ML-generated routing suggestions.

The system is designed to keep human approval in the loop for sensitive decisions such as refunds.

### Key capabilities

- Support request classification and category suggestions.
- Ticket creation and retrieval.
- Order-based refund eligibility assessment.
- Policy-grounded response generation.
- Human-review requirements for refund decisions.
- ML intent and category suggestions with confidence metrics.
- Optional semantic retrieval, with keyword retrieval as the default.
- PostgreSQL persistence with versioned Alembic migrations.
- Interactive API documentation through FastAPI Swagger UI.

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
        |          |
        |          +---- Policy retrieval
        |          +---- ML suggestions
        |          +---- Response generation
        |          +---- Human-review controls
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

The health check verifies application availability. It does not, by itself, prove database connectivity.

## API endpoints

The deployed API includes the following routes. Refer to `/docs` for the authoritative request schemas, HTTP methods, and response models.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Application health check |
| `POST /support/classify` | Classify a support request |
| `POST /tickets` | Create a support ticket |
| `GET /tickets/{ticket_id}` | Retrieve a ticket |
| `GET /orders/{order_id}/refund-assessment` | Assess refund eligibility |
| `POST /support/resolve` | Run the support-resolution workflow |

Some endpoints may require specific request headers or body fields. Use the live Swagger documentation to confirm their current contracts.

## Refund safety and human approval

Refund assessment is separate from refund execution.

A response such as:

```json
{
  "eligible": true,
  "amount_minor": 499900,
  "currency": "INR",
  "reason": "ELIGIBLE_PENDING_HUMAN_APPROVAL",
  "requires_approval": true
}
```

indicates that the order meets the configured demo eligibility checks but still requires human approval.

`amount_minor` represents the amount in the currency's minor units; for INR, `499900` corresponds to ₹4,999.00.

ML predictions are suggestions and should not independently authorize refunds. The deployed demo has been tested for assessment and support-resolution responses; this is not evidence of a live payment-provider integration or actual refund execution.

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

If PowerShell blocks activation, use the appropriate execution-policy setting for your environment or invoke the virtual environment's Python directly.

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

Set `DATABASE_URL` to your Neon PostgreSQL connection string or a suitable local PostgreSQL connection string.

The connection string must be compatible with the installed Psycopg 3 driver. A SQLAlchemy URL using `postgresql+psycopg://` is suitable.

Example format — replace the placeholders locally:

```env
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST/DATABASE?sslmode=require
SUPPORT_RETRIEVAL_MODE=keyword
SUPPORT_LLM_ENABLED=false
GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-120b
```

Use the actual connection string supplied by Neon. Keep `.env` private and never commit credentials to source control.

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

The seed script is intended to be idempotent. Review its contents before using it in any non-demo environment.

### 7. Start the API locally

```powershell
uvicorn app.main:app --reload
```

Open:

- API: http://127.0.0.1:8000
- Swagger UI: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health

## Environment variables

| Variable | Purpose |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `SUPPORT_RETRIEVAL_MODE` | Retrieval mode; use `keyword` for the current deployment |
| `SUPPORT_LLM_ENABLED` | Enables or disables optional LLM functionality |
| `GROQ_API_KEY` | Optional Groq API credential when LLM functionality is enabled |
| `GROQ_MODEL` | Configured Groq model name |
| `PYTHON_VERSION` | Python version selected for the Render build |

Keep secret values in local `.env` or the Render service's environment settings. `.env.example` must contain placeholders only.

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

Configure the required environment variables in the Render dashboard. The `DATABASE_URL` value must point to the Neon database.

The application listens on the port supplied by Render through `$PORT`. Render builds and starts the web service from the connected GitHub repository.

After deployment, verify `/health`, `/docs`, and a database-backed endpoint.

## Database migrations and seed data

Alembic migration files live under `migrations/versions/`.

Apply schema changes with:

```powershell
python -m alembic upgrade head
```

Load the synthetic demo records with:

```powershell
python -m scripts.seed_data
```

Do not run schema migrations or seed scripts automatically on every application startup unless that behavior is deliberately designed and tested.

## Testing

Install the development dependencies if they are maintained separately. The project test suite can be run with:

```powershell
python -m pytest -q
```

Run tests locally before pushing changes. Use a dedicated test database and synthetic data for integration tests.

## Repository structure

The main project areas are:

```text
customer-support-resolution-agent/
├── app/
│   ├── main.py
│   ├── db.py
│   └── agent/
├── migrations/
│   ├── env.py
│   └── versions/
├── scripts/
│   └── seed_data.py
├── tests/
├── requirements.txt
├── requirements-semantic.txt
├── Procfile
├── alembic.ini
├── .env.example
├── .gitignore
└── README.md
```

This is a high-level view of the principal files; the exact contents of subdirectories may evolve as the project develops.

## Security and operational notes

- Never commit `.env`, database passwords, or API keys.
- Use synthetic data in demos and tests.
- Keep refund approval separate from eligibility assessment.
- Restrict and authenticate sensitive endpoints before production use.
- Add appropriate rate limits, structured logging, monitoring, and error handling before handling real customer traffic.
- Review database access permissions and backup requirements.
- Review Render and Neon service limits and current pricing before relying on a free-tier deployment for critical workloads.
- Do not treat a successful health check as a complete database or business-workflow test.

## Current deployment verification

The following checks have succeeded during deployment verification:

- Render health endpoint returned HTTP 200.
- Refund assessment returned an eligibility result requiring human approval.
- Ticket creation returned a generated ticket ID.
- Ticket retrieval returned the created ticket.
- Support resolution returned a structured response with ML suggestions and a human-review outcome.
- The Git working tree was clean, and the `main` branch was synchronized with `origin/main`.

