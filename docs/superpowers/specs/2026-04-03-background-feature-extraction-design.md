# Background Feature Extraction — Design Spec

## Problem

HuggingFace Space API calls for image feature extraction and classification block the AddPetView and ReportPetLocationView requests for 15-30 seconds, degrading UX. Users cannot continue using the site until extraction completes.

## Solution

Move feature extraction to background processing using Django-Q2 with the ORM broker (existing PostgreSQL via Supabase). The pet/location is saved immediately with `feature_status='pending'`, and a background task handles extraction. Users are notified via the existing Notification model when processing completes or fails.

## Constraints

- **Azure App Service B1** — 1 core, 1.75 GB RAM, single instance
- **Azure for Students** — limited budget, no room for Redis/RabbitMQ
- **Supabase PostgreSQL** — used as both database and Q2 broker
- **Frontend on Cloudflare** — API-only changes, no frontend deployment needed

---

## 1. Model Changes

### Pet model — new field

```python
FEATURE_STATUS_CHOICES = [
    ('pending', 'Pending'),
    ('processing', 'Processing'),
    ('completed', 'Completed'),
    ('failed', 'Failed'),
]

feature_status = models.CharField(
    max_length=20, choices=FEATURE_STATUS_CHOICES, default='pending',
    db_index=True
)
```

### PetLocation model — new field + cleanup

```python
feature_status = models.CharField(
    max_length=20, choices=FEATURE_STATUS_CHOICES, default='pending',
    db_index=True
)
```

**Pre-existing cleanup required:**
- PetLocation has duplicate `features` field definitions (lines ~187 and ~248). Consolidate into one: `features = models.JSONField(default=list, validators=[validate_json_list])`.
- PetLocation has duplicate `extract_and_store_features` method definitions (lines ~189-230 and ~286-320). Remove the first, keep the second.
- **Remove the `PetLocation.save()` override** that auto-triggers `extract_and_store_features()` when `image` is set but `features` is empty. This conflicts with async processing — saving with `feature_status='processing'` would re-trigger synchronous extraction. Background tasks will handle extraction instead.

### PetLocationSerializer

Add `feature_status` as read-only field (same as PetSerializer).

### Migration

One migration adding `feature_status` (with `db_index=True`) to both Pet and PetLocation. Default `'pending'` for existing rows.

---

## 2. Django-Q2 Setup

### Dependency

```
django-q2
```

### Settings

```python
INSTALLED_APPS = [
    ...
    'django_q',
]

Q_CLUSTER = {
    'name': 'pawgle',
    'workers': 1,          # Single worker — B1 has only 1.75GB RAM
    'recycle': 50,          # Recycle worker after 50 tasks to prevent memory leaks
    'timeout': 180,         # 3 min per task (HF Space cold starts can take 60s+)
    'retry': 300,           # Retry after 5 min if no result
    'orm': 'default',       # Use PostgreSQL as broker
    'bulk': 10,
    'queue_limit': 50,      # Limit in-memory queue size
    'catch_up': False,
}
```

**Note:** Django-Q2 does not have a `max_attempts` config key. Attempt counting is handled within the task function itself (see Section 3).

### Deployment

Azure startup command changes to a script that runs both qcluster and gunicorn:

```bash
#!/bin/bash
trap 'kill $(jobs -p)' EXIT
python manage.py migrate --noinput
python manage.py qcluster &
gunicorn animal.wsgi --bind=0.0.0.0:8000 -w 1
```

The `trap` ensures qcluster is cleaned up if gunicorn crashes and Azure restarts the container.

---

## 3. Background Tasks (`accounts/tasks.py`)

### `extract_pet_features(pet_id, attempt=1)`

1. Fetch Pet by ID — **guard: if Pet does not exist (deleted), log and return silently**
2. Check `feature_status == 'completed'` → skip (idempotency)
3. Set `feature_status = 'processing'`, save
4. **Download image from `pet.images[0]` URL via `requests.get()`** to temp file (Pet stores URLs as strings in a JSONField list)
5. Call `pawgle_client.extract_features(temp_path)`
6. Call `pawgle_client.classify_pet(temp_path)`
7. Save `features`, `additionalInfo['ai_classification']`, and `feature_status = 'completed'`
8. **Guard: re-fetch Pet to confirm it still exists** before creating Notification
9. Create Notification: verb=`"feature_extraction_complete"`, description=`"Your pet {name}'s features have been extracted successfully"`, target=pet
10. Clean up temp file
11. **On failure:** if `attempt < 3`, re-dispatch with `attempt + 1`. Otherwise set `feature_status = 'failed'` and create failure notification (with Pet existence guard).

### `extract_location_features(pet_location_id, attempt=1)`

1. Fetch PetLocation by ID — guard if deleted
2. Check idempotency
3. Set `feature_status = 'processing'`
4. **Read image via `location.image.read()`** (PetLocation uses Django ImageField with SupabaseStorage — different from Pet's URL list)
5. Write to temp file, extract features
6. Save features + `feature_status = 'completed'`
7. Clean up temp file
8. On failure: same retry logic as above, no notification

### Attempt counting

Django-Q2 has no built-in `max_attempts`. Instead, the `attempt` parameter is passed through re-dispatch:

```python
def extract_pet_features(pet_id, attempt=1):
    try:
        # ... extraction logic ...
    except Exception as e:
        if attempt < 3:
            async_task('accounts.tasks.extract_pet_features', pet_id, attempt + 1)
        else:
            pet.feature_status = 'failed'
            pet.save(update_fields=['feature_status'])
            # create failure notification
```

### Idempotency

Tasks check if `feature_status == 'completed'` before processing and skip if so, preventing duplicate work.

---

## 4. View Changes

### AddPetView

**Before:** Validates, uploads image, blocks on HF calls, saves pet, returns in 15-30s.

**After:**
1. Validate fields + image
2. Upload image to Supabase (~1-2s)
3. Save pet with `feature_status='pending'`, `features=[]`
4. Dispatch `async_task('accounts.tasks.extract_pet_features', pet.id)`
5. Return 201 immediately (~2-3s)

No temp files or HF calls in the request path.

### ReportPetLocationView

Same pattern:
1. Validate + save location with `feature_status='pending'`
2. Dispatch `async_task('accounts.tasks.extract_location_features', location.id)`
3. Return 201 immediately

### SearchPetView

Modified to handle pets without pre-extracted features:
1. Extract features from uploaded search image (synchronous — user is waiting for results)
2. Split database pets into two groups:
   - **Group A:** Pets WITH features (`feature_status='completed'`) → cosine similarity in-memory (fast)
   - **Group B:** Pets WITHOUT features (`feature_status` in `pending/processing/failed`) → download their images via `requests.get()` from stored URLs, call `batch_compare_features` on HF Space
3. Merge results, sort by similarity, return
4. Include `"pending_features_count": N` in response so frontend can inform users if some matches may be incomplete

**Note:** If Group B is large (>10 pets), cap it at 10 to avoid excessive latency and return `"skipped_pending_count"` in the response. This is a pragmatic trade-off — the queue will process remaining pets shortly.

---

## 5. API Changes

### Modified response — AddPetView

```json
{
  "success": true,
  "pet": { "...", "feature_status": "pending" },
  "message": "Pet added successfully. Features are being extracted in the background."
}
```

### New endpoint — Feature status

`GET /api/auth/pets/<int:pet_id>/feature-status/`

```json
{ "feature_status": "pending|processing|completed|failed" }
```

Frontend can poll this or rely on notifications.

### New endpoint — Retry features

`POST /api/auth/pets/<int:pet_id>/retry-features/`

- Owner-only, only when `feature_status = 'failed'`
- Resets to `pending`, dispatches new task

```json
{ "message": "Feature extraction re-queued" }
```

### Serializer changes

- `PetSerializer` — add `feature_status` as read-only field, included in all pet responses.
- `PetLocationSerializer` — add `feature_status` as read-only field.

---

## 6. Notifications

Uses existing Notification model:

| Event | verb | description |
|-------|------|-------------|
| Success | `feature_extraction_complete` | "Your pet {name}'s features have been extracted successfully" |
| Failure | `feature_extraction_failed` | "Feature extraction failed for {name}. You can retry from your profile." |

`target` FK points to the Pet instance.

**Guard:** Before creating a notification, verify the Pet still exists (it may have been deleted between task dispatch and completion). If Pet is gone, skip notification silently.

---

## 7. Error Handling & Retries

- **Automatic:** Task self-retries up to 3 attempts via `attempt` parameter
- **Manual:** Retry endpoint for pet owners when `feature_status = 'failed'`
- **Admin:** Management command `retry_failed_features` re-dispatches all failed tasks
- **App restart:** Tasks persist in PostgreSQL queue; qcluster picks them up on restart
- **HF Space down:** Fails after 3 attempts, user gets failure notification, can retry later
- **Duplicate dispatch:** Task checks `feature_status == 'completed'` and skips
- **Pet deleted mid-task:** Guard checks existence before saving results or creating notifications

---

## 8. Files Changed

| File | Change |
|------|--------|
| `requirements.txt` | Add `django-q2` |
| `animal/settings.py` | Add `django_q` to INSTALLED_APPS, add Q_CLUSTER config |
| `accounts/models.py` | Add `feature_status` to Pet and PetLocation; clean up duplicate `features` field and `extract_and_store_features` method; remove `PetLocation.save()` auto-extraction override |
| `accounts/tasks.py` | **New file** — `extract_pet_features`, `extract_location_features`, retry logic |
| `accounts/views.py` | Simplify AddPetView, ReportPetLocationView; modify SearchPetView for Group A/B; add FeatureStatusView, RetryFeaturesView |
| `accounts/serializers.py` | Add `feature_status` to PetSerializer and PetLocationSerializer |
| `accounts/urls.py` | Add `pets/<int:pet_id>/feature-status/` and `pets/<int:pet_id>/retry-features/` |
| `startup.sh` | **New file** — Azure startup script with trap, running qcluster + gunicorn |
| Migration | Add `feature_status` (indexed) to Pet and PetLocation |
