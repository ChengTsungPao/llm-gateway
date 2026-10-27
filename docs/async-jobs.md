# Async jobs

For background LLM work that must not block or be blocked by a client waiting on a socket, such as
Memory Hub's memory extraction. Code: `llm_gateway/jobs.py`; routes in `server.py`.

## API

| Method | Path | |
|---|---|---|
| POST | `/v1/async/chat/completions` | Chat Completions body plus `priority`, `dedup_key`, `timeout_s`, `max_attempts`. Returns `202` with `job_id` and `status: queued` at once. |
| GET | `/v1/jobs/{id}` | The job; `result` (the Chat Completions response) once `succeeded`. `404` if the job does not exist: treat it as lost and submit again. |
| GET | `/v1/jobs` | List, newest first. Filters: `status=suspended,failed`, `priority`, `key_prefix`, `limit`. |
| GET | `/v1/jobs/stats` | `today_requests`, `today_success`, `today_failed` (counted from `since`, the gateway's local midnight), queued per priority, `waiting` (up to 20 queued jobs in run order), the running job, dead letters. Also in `/api/status` as `jobs`. |
| POST | `/v1/jobs/{id}/resume` | Run a `suspended` or `failed` job again (attempts start over; the same request, not a new one). |
| DELETE | `/v1/jobs/{id}` | Discard a job for good (not while it runs). |
| GET | `/v1/jobs/events` | What happened to dead letters, newest first: `dead_letter`, `resumed`, `recovered`, `discarded`. Filters: `job_id`, `limit`. Kept 90 days, also after the job itself is deleted. |

## Behaviour

- **One job at a time.** Jobs go through the same model scheduler as every request (`manager.py`), so
  a job never switches the model under a running chat or agent task.
- **Priorities.** `high` runs like a chat request; `normal` like an API request (after chats and
  agents); `idle_only` only when no `high` or `normal` job is queued (even one waiting out a backoff)
  and nothing is running, loading or holding the model.
- **Timing.** Waiting in the queue, for the GPU or for a model to load is `queued` and has no timeout.
  The execution timeout (`timeout_s`, default 900 s) starts when generation starts (`running`).
- **Retries.** Timeouts, 429, 5xx, connection errors and "model busy" are retried with an exponential
  backoff (30 s, 60 s, 120 s, ... up to 15 min, with jitter). After `max_attempts` (default 5) the job is
  `suspended`: a dead letter. 400, 401, 403, 404 and 422 are `failed` at once.
- **Counting a dead letter.** It counts as failed on the day it fails. If it is resumed (or submitted
  again) and then succeeds, that same day counts it as a success instead, so every day's requests equal
  its successes, failures and jobs still open. A discarded dead letter stays counted as failed. Each
  step is written to `/v1/jobs/events` and the gateway log; the job keeps `dead_count`.
- **Idempotency.** Submitting a `dedup_key` that exists returns that job instead of a new one. A `failed`
  job submitted again is run again; a `discarded` one stays discarded, so its submitter can tell that
  the user dropped that work.
- **Persistence.** SQLite at `<agent.data_dir>/jobs.db` (`data/jobs.db`, mounted in Docker): the queue,
  the dead letters and today's numbers survive a restart. A job that was running is queued again.
  Succeeded and discarded jobs are deleted after 7 days.

## Settings (`config.json`, all optional)

```json
"jobs": {"max_attempts": 5, "backoff_base_s": 30, "backoff_cap_s": 900, "timeout_s": 900}
```

## UI

The Models page shows today's requests, successes, failures and the job queue (queued jobs plus the one running) as one row of cards, with
the time the counts start from (the gateway's midnight: set `TZ` in `.env`, e.g. `TZ=Asia/Taipei`, or
it is UTC midnight in Docker). Queued background jobs are listed under the running job in the
Resources panel, in the order they will run. The dead letters are listed with **Resume** and
**Discard** (asks once to confirm).
