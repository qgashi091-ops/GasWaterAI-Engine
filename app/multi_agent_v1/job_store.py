"""Job-record storage for the asynchronous Multi-Agent v1 job mode (see
job_manager.py and POST/GET /multi_agent_v1/jobs in app/main.py).

PRODUCTION BOUNDARY -- read before deploying or extending this:

`InMemoryJobStore` below is a POC implementation only. It keeps every job
record in a plain Python dict inside this process's memory. On Render Free
(and Starter, without a paid persistent disk/add-on), that means:

  - A job record is LOST if the process restarts or spins down for any
    reason (deploy, crash, Render's own idle-timeout on the free tier,
    an out-of-memory kill) while that job is queued, processing, or still
    within its TTL window. A client polling GET /multi_agent_v1/jobs/{id}
    after such a restart gets 404, not a stale-but-truthful answer --
    there is no durable store to answer from.
  - State is NOT shared across multiple process instances. If the hosting
    platform ever runs more than one web worker/instance of this app,
    each instance has its OWN job table; a job created against instance A
    cannot be polled from instance B. This POC assumes exactly one
    running instance, which matches today's single Render Free/Starter
    web service.
  - Nothing here is written to disk at all (deliberately -- see section 6
    of the job-mode request: never write uploaded PDFs or plan content to
    local storage), so there is no local-file persistence to fall back on
    either.

None of this is hidden behind a falsely reassuring name or interface: the
class is named `InMemoryJobStore`, not `JobStore`, and `JobStore` itself
(the abstract interface below) says nothing about durability -- a future
swap to a real persistent backend only needs a new class implementing the
same four methods.

What a truly robust, production-grade job store would require, since
Render Free/Starter offers no reliable persistent local filesystem and no
in-process state survives a restart or a second instance: an EXTERNAL
component shared by every instance and surviving process restarts --
for example a managed Redis (Render's own Key Value add-on, a small
Upstash/Redis Cloud instance) for the hot queue/status lookups, or a
managed Postgres (Render Postgres, already a common pairing) for the job
table itself, with uploaded PDF bytes (while a job is in flight) kept
client-side/re-sent or in object storage (e.g. S3-compatible), never
assumed to still be on local disk after a restart. That is a real
infrastructure decision (a new paid component, credentials, a migration)
deliberately NOT made here -- this POC only establishes the `JobStore`
seam so such a backend can be substituted later without touching
job_manager.py or app/main.py.
"""
from __future__ import annotations

import copy
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

QUEUED = "queued"
PROCESSING = "processing"
COMPLETED = "completed"
FAILED = "failed"

ACTIVE_STATUSES = (QUEUED, PROCESSING)
TERMINAL_STATUSES = (COMPLETED, FAILED)


@dataclass
class JobRecord:
    job_id: str
    fingerprint: str
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[dict] = None
    error: Optional[str] = None


class JobStore(ABC):
    """Storage seam for job records. `InMemoryJobStore` is the only POC
    implementation today; a production deployment would implement this
    same interface against an external store (see module docstring)."""

    @abstractmethod
    def create(self, record: JobRecord) -> None: ...

    @abstractmethod
    def get(self, job_id: str) -> Optional[JobRecord]: ...

    @abstractmethod
    def update(self, job_id: str, **fields) -> None: ...

    @abstractmethod
    def find_active_by_fingerprint(self, fingerprint: str) -> Optional[JobRecord]:
        """Returns a QUEUED or PROCESSING job for this fingerprint, if one
        exists -- the idempotency check (section 7 of the job-mode spec):
        a second identical upload while one is already in flight must
        reuse that job_id, never start a second, costly analysis."""

    @abstractmethod
    def delete_completed_before(self, cutoff: datetime) -> int:
        """Removes every COMPLETED/FAILED record whose completed_at is
        older than `cutoff` (TTL cleanup, section 6). Never removes a
        QUEUED/PROCESSING job regardless of age. Returns the count
        removed."""


class InMemoryJobStore(JobStore):
    """Thread-safe POC JobStore -- see this module's own docstring for the
    explicit production-persistence boundary this implementation does NOT
    cross."""

    def __init__(self) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def create(self, record: JobRecord) -> None:
        with self._lock:
            self._jobs[record.job_id] = copy.copy(record)

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            record = self._jobs.get(job_id)
            return copy.copy(record) if record is not None else None

    def update(self, job_id: str, **fields) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            for key, value in fields.items():
                setattr(record, key, value)

    def find_active_by_fingerprint(self, fingerprint: str) -> Optional[JobRecord]:
        with self._lock:
            for record in self._jobs.values():
                if record.fingerprint == fingerprint and record.status in ACTIVE_STATUSES:
                    return copy.copy(record)
            return None

    def delete_completed_before(self, cutoff: datetime) -> int:
        with self._lock:
            to_delete = [
                job_id for job_id, record in self._jobs.items()
                if record.status in TERMINAL_STATUSES
                and record.completed_at is not None
                and record.completed_at < cutoff
            ]
            for job_id in to_delete:
                del self._jobs[job_id]
            return len(to_delete)
