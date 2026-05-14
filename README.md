# FlowForge

**FlowForge** is an AI-powered project planning and diagram generation platform. Feed it a client proposal and it runs a multi-agent pipeline to produce project timelines, Gantt charts, and technical architecture diagrams — all validated before delivery.

---

## Table of Contents

- [Features](#features)
- [System Architecture](#system-architecture)
- [Agent Workflow](#agent-workflow)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running the API](#running-the-api)
- [API Usage](#api-usage)
- [Diagram Types](#diagram-types)
- [Database](#database)
- [Running Tests](#running-tests)
- [CI/CD](#cicd)

---

## Features

- **Multi-agent pipeline** — Specialized agents for timeline generation, planning, diagram creation, and validation
- **Self-healing loops** — Invalid outputs are automatically retried with corrective feedback (up to 3 retries per stage)
- **Prompt optimization** — LangChain-powered prompt refinement before agent execution
- **6 diagram types** — Workflow, CI/CD, System Design, Flowchart, Architecture, Gantt
- **Persistent sessions** — PostgreSQL-backed session tracking with full output versioning
- **FastAPI REST API** — Single endpoint to trigger the full pipeline

---

## System Architecture

```mermaid
graph TB
    subgraph Client
        A[HTTP Client]
    end

    subgraph API["FastAPI Layer (src/api)"]
        B[POST /api/v1/generate-diagrams]
        C[GET /api/v1/health]
    end

    subgraph Pipeline["Pipeline (src/pipeline)"]
        D[PromptOptimizer\nLangChain]
    end

    subgraph Agents["Agent Layer (src/agents)"]
        E[TimeAgent]
        F[TimeValidatorAgent]
        G[PlanAgent]
        H[ImageGeneratorAgent]
        I[ValidatorAgent]
    end

    subgraph Workflow["LangGraph Workflow (src/workflow)"]
        J[StateGraph\nFlowForgeState]
    end

    subgraph DB["Database Layer (src/database)"]
        K[SessionManager]
        L[(PostgreSQL)]
        M[Sessions Table]
        N[SessionOutputs Table]
        O[TimeValidationSessions Table]
    end

    subgraph External["External Services"]
        P[HuggingFace\nInference API]
        Q[Together AI\nLLM Provider]
    end

    A -->|JSON Request| B
    B --> D
    D --> J
    J --> E
    J --> F
    J --> G
    J --> H
    J --> I
    E & F & G & H & I -->|Read/Write State| J
    E & F & G & H & I -->|Persist Outputs| K
    K --> L
    L --- M & N & O
    E & G & H -->|LLM Calls| P
    D -->|Prompt Optimization| Q
```

---

## Agent Workflow

```mermaid
flowchart TD
    START([Client Request]) --> OPT[Prompt Optimizer\nLangChain refinement]
    OPT --> TA[TimeAgent\nMilestones · Gantt · Parallel Streams]

    TA -->|error| END_ERR([End — Error])
    TA -->|ok| TVA[TimeValidatorAgent\nValidates timeline structure]

    TVA -->|valid| PA[PlanAgent\nTask breakdown · Resources · Risks]
    TVA -->|invalid & retries left| TA
    TVA -->|max retries reached| END_ERR

    PA -->|error| END_ERR
    PA -->|ok| IGA[ImageGeneratorAgent\nGenerates Mermaid diagrams]

    IGA -->|error| END_ERR
    IGA -->|ok| VA[ValidatorAgent\nValidates all diagrams]

    VA -->|all valid| END_OK([End — Success\nDiagrams + Timeline + Plan])
    VA -->|invalid & retries left| IGA
    VA -->|max retries reached| END_OK

    style START fill:#4CAF50,color:#fff
    style END_OK fill:#2196F3,color:#fff
    style END_ERR fill:#f44336,color:#fff
    style OPT fill:#FF9800,color:#fff
    style TA fill:#9C27B0,color:#fff
    style TVA fill:#9C27B0,color:#fff
    style PA fill:#9C27B0,color:#fff
    style IGA fill:#9C27B0,color:#fff
    style VA fill:#9C27B0,color:#fff
```

**Retry limits:** 3 retries for timeline validation, 3 retries for diagram validation. After hitting the limit the workflow exits with whatever valid output it has.

---

## Project Structure

```
FlowForge/
├── main.py                     # Uvicorn entry point
├── requirements.txt
├── alembic.ini
├── alembic/                    # Database migrations
│   └── versions/
├── src/
│   ├── app.py                  # FastAPI app + CORS
│   ├── config.py               # Env config (HF_TOKEN, DATABASE_URL)
│   ├── logger.py
│   ├── api/
│   │   ├── router.py           # API endpoints
│   │   └── dependencies.py     # HF token injection
│   ├── agents/
│   │   ├── base_agent.py       # Abstract base class
│   │   ├── time_agent.py       # Timeline + Gantt generation
│   │   ├── time_validator_agent.py
│   │   ├── plan_agent.py       # Project plan generation
│   │   ├── image_generator_agent.py  # Mermaid diagram generation
│   │   └── validator_agent.py  # Diagram validation
│   ├── pipeline/
│   │   └── prompt_optimizer.py # LangChain prompt optimization
│   ├── workflow/
│   │   └── graph_workflow.py   # LangGraph StateGraph definition
│   ├── schemas/
│   │   ├── request.py          # Pydantic request models
│   │   └── response.py         # Pydantic response models
│   ├── schema/
│   │   └── helpers.py          # Response formatting
│   ├── database/
│   │   ├── models.py           # SQLAlchemy models
│   │   ├── session_manager.py  # Session CRUD
│   │   ├── service.py          # DB service layer
│   │   └── base.py             # Declarative base
│   └── utils/
│       ├── retry.py            # Retry utilities
│       └── decode.py
└── tests/
    ├── test_units.py
    └── test_integration.py
```

---

## Prerequisites

- Python 3.11+
- PostgreSQL (for session persistence)
- [HuggingFace account](https://huggingface.co) with an API token

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/your-org/FlowForge.git
cd FlowForge

# 2. Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and fill in environment variables
cp .env.example .env   # or edit .env directly

# 5. Run database migrations
alembic upgrade head
```

---

## Configuration

Create a `.env` file in the project root:

```env
HF_TOKEN=hf_your_huggingface_token_here
DATABASE_URL=postgresql://user:password@localhost:5432/flowforge
```

| Variable       | Required | Description                              |
|----------------|----------|------------------------------------------|
| `HF_TOKEN`     | Yes      | HuggingFace API token for model inference |
| `DATABASE_URL` | Yes      | PostgreSQL connection string             |

---

## Running the API

```bash
python main.py
```

The API starts at `http://localhost:8000`. Interactive docs are available at `http://localhost:8000/docs`.

---

## API Usage

### Generate Diagrams

**`POST /api/v1/generate-diagrams`**

```bash
curl -X POST http://localhost:8000/api/v1/generate-diagrams \
  -H "Content-Type: application/json" \
  -d '{
    "proposal": {
      "title": "E-Commerce Platform",
      "description": "Build a scalable online store with inventory management and payment processing.",
      "requirements": [
        "User authentication and authorization",
        "Product catalog with search",
        "Shopping cart and checkout",
        "Payment gateway integration"
      ],
      "constraints": ["Must be GDPR compliant", "Mobile-first design"],
      "timeline_weeks": 16,
      "team_size": 6,
      "tech_stack": ["React", "FastAPI", "PostgreSQL", "Redis"],
      "budget_range": "$80k-$120k"
    },
    "prompt": {
      "user_prompt": "Generate a complete system architecture and CI/CD pipeline for this e-commerce platform.",
      "diagram_types": ["workflow", "ci_cd", "system_design"],
      "optimize_prompt": true,
      "priority": "high"
    },
    "hf_token": "hf_your_token_here"
  }'
```

**Response shape:**

```json
{
  "status": "success",
  "proposal_summary": "E-Commerce Platform...",
  "optimized_prompt": "...",
  "timeline": {
    "milestones": ["..."],
    "parallel_work_streams": ["..."],
    "gantt_chart": "gantt\n  title ..."
  },
  "plan": {
    "task_breakdown": "...",
    "resource_allocation": "...",
    "risk_mitigation": "..."
  },
  "diagrams": [
    {
      "diagram_type": "workflow",
      "mermaid_code": "flowchart TD\n  ...",
      "title": "System Workflow",
      "is_valid": true
    }
  ],
  "overall_validation": true
}
```

### Health Check

**`GET /api/v1/health`**

```bash
curl http://localhost:8000/api/v1/health
```

---

## Diagram Types

| Type            | Value           | Description                              |
|-----------------|-----------------|------------------------------------------|
| Workflow        | `workflow`      | End-to-end process flow                  |
| CI/CD Pipeline  | `ci_cd`         | Build, test, and deploy pipeline         |
| System Design   | `system_design` | High-level system components             |
| Flowchart       | `flowchart`     | Decision and process flowchart           |
| Architecture    | `architecture`  | Technical architecture overview          |
| Gantt Chart     | `gantt`         | Project timeline (auto-included)         |

---

## Database

FlowForge uses PostgreSQL with Alembic for migrations. Three tables are created:

- **`sessions`** — One row per workflow run, stores all input parameters
- **`session_outputs`** — Versioned agent outputs with validation status and feedback
- **`time_validation_sessions`** — Timeline validation history with retry tracking

```bash
# Create a new migration after model changes
alembic revision --autogenerate -m "describe your change"

# Apply migrations
alembic upgrade head

# Roll back one step
alembic downgrade -1
```

See [DATABASE.md](DATABASE.md) for connection pooling details.

---

## Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/test_units.py

# Integration tests only
pytest tests/test_integration.py

# With coverage
pytest --cov=src tests/
```

---

## CI/CD

GitHub Actions workflows are defined in `.github/workflows/`:

| Workflow     | Trigger                        | Purpose                        |
|--------------|--------------------------------|--------------------------------|
| `dev.yml`    | Push to `dev`                  | Run tests on dev branch        |
| `dev-pr.yml` | PR targeting `dev`             | Lint + test on pull requests   |
| `staging.yml`| Push to `staging`              | Deploy to staging environment  |
