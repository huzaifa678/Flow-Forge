# FlowForge Session Database Setup

## Environment Variables

```bash
export DATABASE_URL="postgresql://user:password@localhost/flowforge"
```

## Connection Pooling

The database layer uses SQLAlchemy with QueuePool:
- Pool size: 10 connections
- Max overflow: 20 connections
- Pre-ping: enabled (validates connections before use)
- Recycle: 3600 seconds (1 hour)

## Alembic Commands

```bash
# Initialize migrations (already done)
alembic revision --autogenerate -m "description"

# Run migrations
alembic upgrade head

# Rollback
alembic downgrade -1

# Show history
alembic history --verbose
```

## Usage

The workflow automatically runs migrations on startup:

```python
from src.workflow.graph_workflow import run_flowforge_workflow

result = run_flowforge_workflow(
    proposal="Build a web application",
    prompt="Detailed requirements...",
    hf_token="your-token"
)
```

Sessions are created automatically and agent outputs are stored with version tracking. Use `session_manager.get_previous_invalid_output()` to retrieve failed outputs for improvement prompts.

## Cleanup

Call `close_db()` when shutting down to properly dispose of the connection pool.