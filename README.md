# Techvanta Project Hub — Updated

This version keeps the team login system, Research/Knowledge, Resources and project activity, while replacing the research-bot workflow with team/project management.

## Important: fix for `public.tasks` not found
The error:

`PGRST205: Could not find the table 'public.tasks' in the schema cache`

means the Supabase database did not have the new project-management table. **Run `schema.sql` in Supabase SQL Editor.** Do not skip this step.

The SQL file safely uses `CREATE TABLE IF NOT EXISTS` and `ALTER TABLE ... ADD COLUMN IF NOT EXISTS`, so it is designed as a migration for the existing Techvanta database.

## New project-management features
- Team members and roles
- Tasks, assignments, priorities and deadlines
- Kanban board
- Milestones
- Shared Research / useful SIH26090 knowledge
- Project resources
- Activity tracking
- AI-ranked contribution leaderboard
- Native WebSocket Team Chat
- Whole-team 24-hour chat reset
- SIH26090-restricted Groq Project AI Assistant

## Leaderboard logic
The leaderboard is no longer based mainly on time.

The backend measures real evidence from the database, including:
- assigned and completed tasks
- completion rate
- on-time delivery
- overdue work
- high/urgent-priority work completed
- task completion speed (quick working)
- role/responsibility
- research added
- useful research
- research upgrades
- research quality score
- milestones created and milestone progress
- project resources added
- recent active participation
- workspace time

These measured facts are sent to Groq. Groq returns the ranking, score and explanation. Time is only one factor. If Groq is unavailable or returns invalid JSON, a deterministic evidence-based fallback is used so the leaderboard still works.

## Team Chat
Team Chat is **not the AI chat**.

- Native WebSocket endpoint: `/ws/chat`
- Messages are stored in `team_chat_messages`.
- A shared `team_chat_windows` record defines the current 24-hour window.
- When the 24-hour window expires, the backend deletes the whole previous team chat and starts a new window.
- Connected browsers receive a WebSocket `reset` event immediately when a reset occurs.
- HTTP fallback is retained for environments where WebSocket is unavailable.

For Render/Gunicorn, use a threaded worker so WebSocket connections can remain open:

```bash
gunicorn --worker-class gthread --threads 100 --bind 0.0.0.0:$PORT app:app
```

## Groq AI Assistant
The AI assistant is intentionally locked to:
- SIH26090
- artisan/handmade products
- artisan challenges and market linkage
- relevant government schemes/marketplaces
- the team's research/resources
- the team's implementation, architecture, technology, testing and presentation

Unrelated questions are refused.

The AI context is bounded before being sent to Groq to avoid the earlier HTTP 413 request-size problem.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`.

## Environment

Copy `.env.example` to `.env` and set:

```text
SUPABASE_URL=...
SUPABASE_KEY=...        # server-side service-role key
GROQ_API_KEY=...
GROQ_MODEL=groq/compound
TECHVANTA_TEAM_NAME=Techvanta
TECHVANTA_TEAM_PASSWORD=...
SECRET_KEY=...
```

Never expose `SUPABASE_KEY` or `GROQ_API_KEY` in frontend JavaScript.
