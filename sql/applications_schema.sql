-- ============================================================================
-- sql/applications_schema.sql — Live Job Finder + auto-apply storage
-- Database: data/job_applications.db (created automatically by src/applications.py)
-- ============================================================================

-- Single table: stores collected jobs (dedup by URL) and tracks which jobs
-- have already been applied to, so no job is ever applied to twice.
CREATE TABLE IF NOT EXISTS live_jobs (
    job_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    region       TEXT,                       -- Europe / Middle East / ... / Aerospace
    title        TEXT,                       -- job title
    url          TEXT NOT NULL UNIQUE,       -- unique posting URL (dedup key)
    snippet      TEXT,                       -- snippet from search
    field        TEXT,                       -- DS/ML/DL | Aerospace | Other
    email        TEXT,                       -- contact email (extracted, editable)
    collected_at TEXT,                       -- ISO timestamp
    applied      INTEGER NOT NULL DEFAULT 0, -- 0 = not applied, 1 = applied
    applied_at   TEXT                        -- when it was applied (ISO timestamp)
);

-- Useful queries -------------------------------------------------------------
-- Jobs not yet applied to (candidates for auto-apply):
--   SELECT * FROM live_jobs WHERE applied = 0 ORDER BY job_id DESC;

-- Jobs already applied to (dedup — never re-apply):
--   SELECT * FROM live_jobs WHERE applied = 1 ORDER BY applied_at DESC;

-- Count applied vs. total:
--   SELECT COUNT(*) AS total,
--          SUM(applied) AS applied
--   FROM live_jobs;
