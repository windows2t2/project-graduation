-- ============================================================================
-- sql/schema.sql — v4: Job Market Intelligence Database Schema
-- Run with: sqlite3 job_market.db < sql/schema.sql
-- ============================================================================

-- Drop existing tables (for re-runs)
DROP TABLE IF EXISTS jobs;
DROP TABLE IF EXISTS companies;
DROP TABLE IF EXISTS locations;

-- ---------------------------------------------------------------------------
-- Locations table (normalized)
-- ---------------------------------------------------------------------------
CREATE TABLE locations (
    location_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    country_code  TEXT NOT NULL UNIQUE,
    country_name  TEXT,
    region        TEXT
);

-- ---------------------------------------------------------------------------
-- Companies table
-- ---------------------------------------------------------------------------
CREATE TABLE companies (
    company_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    company_size  TEXT CHECK (company_size IN ('S', 'M', 'L')) NOT NULL,
    location_id   INTEGER REFERENCES locations(location_id)
);

-- ---------------------------------------------------------------------------
-- Jobs table (main fact table)
-- ---------------------------------------------------------------------------
CREATE TABLE jobs (
    job_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    work_year         INTEGER NOT NULL,
    job_title         TEXT NOT NULL,
    experience_level  TEXT CHECK (experience_level IN ('EN', 'MI', 'SE', 'EX')),
    employment_type   TEXT,
    salary_in_usd     REAL NOT NULL,
    remote_ratio      INTEGER CHECK (remote_ratio BETWEEN 0 AND 100),
    company_id        INTEGER REFERENCES companies(company_id),
    employee_location_id INTEGER REFERENCES locations(location_id)
);

-- ---------------------------------------------------------------------------
-- Sample queries for analysis
-- ---------------------------------------------------------------------------

-- Average salary by experience level
SELECT j.experience_level,
       ROUND(AVG(j.salary_in_usd), 0) AS avg_salary,
       COUNT(*) AS job_count
FROM jobs j
GROUP BY j.experience_level
ORDER BY avg_salary DESC;

-- Top 10 highest-paying job titles
SELECT j.job_title,
       ROUND(AVG(j.salary_in_usd), 0) AS avg_salary,
       COUNT(*) AS count
FROM jobs j
GROUP BY j.job_title
ORDER BY avg_salary DESC
LIMIT 10;

-- Salary by company size
SELECT c.company_size,
       ROUND(AVG(j.salary_in_usd), 0) AS avg_salary,
       COUNT(*) AS job_count
FROM jobs j
JOIN companies c ON j.company_id = c.company_id
GROUP BY c.company_size
ORDER BY avg_salary DESC;

-- Remote work trends by year
SELECT j.work_year,
       CASE
           WHEN j.remote_ratio = 0 THEN 'On-site'
           WHEN j.remote_ratio = 50 THEN 'Hybrid'
           WHEN j.remote_ratio = 100 THEN 'Fully Remote'
       END AS remote_type,
       COUNT(*) AS count
FROM jobs j
GROUP BY j.work_year, remote_type
ORDER BY j.work_year, remote_type DESC;
