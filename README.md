# PawGle Backend

Django REST API that powers [PawGle](https://pawgle.neokit.app) — a community platform for reuniting lost pets with their owners through AI-powered image recognition, geographic alerts, and in-app chat.

**Live API:** `https://pawglebackend.neokit.app`
**Frontend repo:** [WhoamiI00/PawGleFrontend](https://github.com/WhoamiI00/PawGleFrontend)

---

## What it does

- **Pet recognition** — pets are registered with multiple photos. The model extracts a 512-dim ArcFace embedding for each image; embeddings are averaged + L2-normalized for a single per-pet vector stored in [Qdrant](https://qdrant.tech/).
- **Image search** — uploads a query photo, fetches the nearest neighbours from Qdrant, hydrates from TiDB.
- **Lost / found map** — geo-tagged `PetLocation` reports.
- **Auto-match on report** — when a "found" report is filed, the system queries Qdrant + geo filter for likely-matching registered pets and opens an in-app chat with the owner.
- **In-app chat** — text + image attachments, 5s polling. Email is only used as a "you have a new message, click to open" pager.
- **QR pet tags** — owners download a printable QR per pet; a scan lands strangers on a public `/found/<animal_id>` page that funnels them into the report + chat flow without revealing the owner's contact details.
- **Nearby alerts** — users save areas of interest; a new lost/found report inside the radius pings them.
- **Auth** — JWT with the refresh token in an httpOnly cookie (XSS-proof), Google OAuth, and passwordless magic-link sign-in.

---

## Stack

| Layer | Tech |
|---|---|
| Framework | Django 5.x + Django REST Framework |
| Async tasks | django-q2 |
| Database | [TiDB Cloud](https://www.pingcap.com/tidb-cloud/) (MySQL wire protocol) |
| Vector search | [Qdrant Cloud](https://cloud.qdrant.io) |
| Object storage | [Cloudflare R2](https://www.cloudflare.com/developer-platform/r2/) (S3-compatible) |
| ML model | ArcFace hosted on a [HuggingFace Space](https://huggingface.co/spaces), invoked via the Gradio client |
| Email | [Resend](https://resend.com/) HTTP API |
| Static files | [WhiteNoise](http://whitenoise.evans.io/) |
| Error tracking | [Sentry](https://sentry.io) |
| Host | Azure App Service for Linux (gunicorn) |
| Python | 3.10+ |
| Package management | [uv](https://docs.astral.sh/uv/) |

---

## Architecture

```
┌────────────┐   HTTPS    ┌───────────────────┐    upserts    ┌───────────┐
│  Frontend  │──────────▶│  Django API       │──────────────▶│  Qdrant   │
│  (Next.js) │           │  (gunicorn)       │               │  vectors  │
└────────────┘           │                   │               └───────────┘
                         │  ┌─────────────┐  │
                         │  │ django-q    │  │   embedding   ┌───────────┐
                         │  │ workers     │──┼──────────────▶│ HuggingFace│
                         │  └─────────────┘  │               │ ArcFace   │
                         └─────┬─────────────┘               └───────────┘
                               │  rows           images
                               ▼               ─────▶ Cloudflare R2
                         ┌───────────┐
                         │  TiDB     │
                         │  (MySQL)  │
                         └───────────┘
```

TiDB is the source of truth. Qdrant holds embeddings + minimal payload (`pet_id`, `owner_id`, `lat`, `lon`) for filtering during search — never PII. Rows in TiDB stay in sync via signals on `Pet` / `PetLocation` deletes.

---

## Local development

### 1. Clone + install

```bash
git clone https://github.com/WhoamiI00/PawGleBackend.git
cd PawGleBackend
uv sync                  # creates .venv and installs from uv.lock
```

If you prefer pip: `pip install -r requirements.txt`. The file is regenerated from `uv.lock` on every dep change.

### 2. Configure `.env`

Copy this template and fill in:

```env
# Django
DEBUG=true

# Database (TiDB Cloud)
TIDB_HOST=...
TIDB_PORT=4000
TIDB_USERNAME=...
TIDB_PASSWORD=...
TIDB_DATABASE=pawgle

# Cloudflare R2 (S3-compatible)
CLOUDFLARE_R2_ACCESS_KEY_ID=...
CLOUDFLARE_R2_ACCESS_KEY=...
CLOUDFLARE_BUCKET_NAME=pawgle
CLOUDFLARE_S3_API=https://<account>.r2.cloudflarestorage.com
R2_PUBLIC_URL=https://pub-<id>.r2.dev

# Qdrant Cloud
QDRANT_CLUSTER_ENDPOINT=https://<id>.cloud.qdrant.io:6333
QDRANT_API_KEY=...
QDRANT_VECTOR_DIM=512                       # optional, defaults shown
QDRANT_PETS_COLLECTION=pawgle_pets
QDRANT_REPORTS_COLLECTION=pawgle_reports

# Auto-match tuning
AUTOMATCH_MIN_SIMILARITY=0.6
AUTOMATCH_MAX_DISTANCE_KM=50

# HuggingFace model
HUGGINGFACE_API_TOKEN=hf_...

# OAuth
GOOGLE_OAUTH_CLIENT_ID=...

# Email (Resend)
DEFAULT_FROM_EMAIL="PawGle <onboarding@resend.dev>"
REPLIES_EMAIL="PawGle Replies <replies@your-domain.app>"

# Frontend URL (used in email magic links + QR codes)
FRONTEND_URL=http://localhost:3000

# CORS / cookie config — dev defaults work for localhost
CORS_ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
REFRESH_COOKIE_SECURE=false                 # set true in prod
REFRESH_COOKIE_SAMESITE=Lax                 # set None in cross-site prod

# Sentry (optional — leave blank to disable)
SENTRY_DSN=
SENTRY_ENVIRONMENT=development
```

### 3. Migrate + run

```bash
uv run python manage.py migrate
uv run python manage.py sync_qdrant         # one-time backfill of existing pets
uv run python manage.py runserver           # http://localhost:8000

# In a second terminal — async task worker
uv run python manage.py qcluster
```

---

## Project layout

```
accounts/
  models.py            — Pet, PetLocation, Conversation, Message, PetMatch,
                         AlertSubscription, Notification, EditedPetImage
  views.py             — auth, pet CRUD, search, report, notifications
  chat_views.py        — in-app messaging endpoints
  tag_views.py         — QR generation + public found-pet landing
  magic_link.py        — passwordless sign-in (signed token)
  alerts.py            — geographic radius alert subscriptions + dispatcher
  automatch.py         — Qdrant-driven lost↔found matcher
  qdrant_index.py      — Qdrant client wrapper with retries
  cookie_auth.py       — httpOnly refresh cookie helpers
  tasks.py             — django-q background tasks (feature extraction etc.)
  signals.py           — keep Qdrant in sync with Pet/PetLocation deletes
  pawgle_client.py     — HuggingFace Gradio client
  storage.py           — Cloudflare R2 storage backend
  management/commands/
    sync_qdrant.py     — backfill embeddings from TiDB to Qdrant
animal/
  settings.py
  urls.py
```

---

## API surface (high level)

All routes are under `/api/auth/` unless noted; the prefix is historical.

### Auth
| Method | Path | Purpose |
|---|---|---|
| POST | `/api/auth/signup/` | Register |
| POST | `/api/auth/login/` | Email + password login |
| POST | `/api/auth/google/` | Google OAuth ID token exchange |
| POST | `/api/auth/magic/request/` | Email a magic sign-in link |
| POST | `/api/auth/magic/consume/` | Redeem a magic-link token |
| POST | `/api/auth/logout/` | Clear refresh cookie |
| POST | `/api/token/refresh/` | Refresh access token (reads cookie) |

### Pets
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/auth/profile/` | Current user + their pets |
| POST | `/api/auth/pets/add/` | Register a pet (multi-photo) |
| PUT | `/api/auth/pets/<id>/edit/` | Edit a pet |
| DELETE | `/api/auth/pets/<id>/delete/` | Delete a pet |
| POST | `/api/auth/pets/search/` | Image similarity search |
| GET | `/api/auth/pets/<id>/qr/` | Download a printable QR tag |
| GET | `/api/auth/found/<animal_id>/` | Public landing (no auth) for QR scans |

### Locations
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/auth/pets/locations/` | All active reports |
| GET | `/api/auth/pets/lost/locations/` | Lost only |
| GET | `/api/auth/pets/found/locations/` | Found only |
| POST | `/api/auth/pets/report/` | File a lost/found report |
| PUT | `/api/auth/pets/locations/<id>/status/` | Resolve / re-open |

### Chat
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/auth/chat/conversations/` | My conversations |
| POST | `/api/auth/chat/conversations/start/` | Open a chat for a `pet_location_id` |
| GET | `/api/auth/chat/conversations/<uuid>/messages/?after=<iso>` | Poll for new messages |
| POST | `/api/auth/chat/conversations/<uuid>/messages/` | Send a message + attachments |

### Alerts
| Method | Path | Purpose |
|---|---|---|
| GET/POST | `/api/auth/alerts/` | List or create a radius alert |
| GET/PATCH/DELETE | `/api/auth/alerts/<id>/` | Manage one alert |

### Notifications
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/auth/notifications/` | List + unread count |
| PUT | `/api/auth/notifications/<id>/read/` | Mark one read |
| PUT | `/api/auth/notifications/read-all/` | Mark all read |

---

## Rate limiting

DRF scoped throttles are on for abuse-prone endpoints:

| Scope | Limit |
|---|---|
| `login` | 10/min |
| `signup` | 5/min |
| `password_reset` | 5/hour |
| `email_verification` | 5/hour |
| `pet_search` | 20/min |
| `pet_write` | 30/min |
| `report_pet` | 10/min |
| `anon` (default) | 60/min |
| `user` (default) | 300/min |

Adjust in `animal/settings.py → REST_FRAMEWORK['DEFAULT_THROTTLE_RATES']`.

---

## Background tasks

Run with `python manage.py qcluster`. Three async tasks today:

1. **`extract_pet_features(pet_id)`** — downloads pet images, calls HuggingFace, mean-pools embeddings, upserts to Qdrant.
2. **`extract_location_features(location_id)`** — same for `PetLocation` reports, then enqueues:
3. **`run_auto_match(location_id)`** + **`run_nearby_alerts(location_id)`** — fire after extraction lands.

All have retry budgets + Sentry capture on final failure.

---

## Deployment

Hosted on **Azure App Service for Linux**. The repo's `startup.sh` runs `migrate`, starts the qcluster worker in the background, and launches gunicorn:

```sh
#!/bin/bash
python manage.py migrate --noinput
python manage.py qcluster &
gunicorn animal.wsgi --bind=0.0.0.0:${PORT:-8000} --timeout 600 -w 1
```

Production env additions:

```env
REFRESH_COOKIE_SECURE=true
REFRESH_COOKIE_SAMESITE=Lax
REFRESH_COOKIE_DOMAIN=.neokit.app          # share cookie across subdomains
CORS_ALLOWED_ORIGINS=https://pawgle.neokit.app
SENTRY_DSN=https://...
SENTRY_ENVIRONMENT=production
SITE_URL=https://pawgle.neokit.app
```

`requirements.txt` is what Azure's Oryx pipeline installs from — it's auto-generated from `uv.lock`, so re-run after `uv add`:

```bash
uv export --format requirements-txt --no-hashes --no-emit-project --no-dev > requirements.txt
```

---

## Migrations + ops notes

- TiDB Cloud drops idle connections after ~60–90s. `CONN_MAX_AGE=60` + `CONN_HEALTH_CHECKS=True` in settings handles reconnection transparently.
- Qdrant collections are auto-created on first upsert. Use `python manage.py sync_qdrant` to backfill after schema changes.
- The Django admin lives at `/admin/`; create a superuser with `python manage.py createsuperuser`.

---

## Contributing

1. Fork + branch from `master`
2. `uv sync` and bring up `.env` per the template above
3. Run `python manage.py check` and the existing happy-path manually before pushing
4. Open a PR with a clear summary

Issues and PRs welcome on [WhoamiI00/PawGleBackend](https://github.com/WhoamiI00/PawGleBackend).

---

## License

This project is part of PawGle. See the umbrella repo for license details.
