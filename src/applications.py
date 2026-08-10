"""
src/applications.py — Local SQLite storage for live jobs + auto-apply tracking.

Stores collected jobs in a local SQLite database so jobs are remembered across
sessions, and tracks which jobs have already been applied to — a job is never
applied to twice (dedup by URL).

Email sending uses only the Python standard library (smtplib + email).
SMTP settings come from `.env`:
    SMTP_HOST=...
    SMTP_PORT=587
    SMTP_USER=...
    SMTP_PASS=...
    EMAIL_FROM=you@example.com
"""
import os
import re
import smtplib
import sqlite3
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path

import pandas as pd

from src.utils import PROJECT_ROOT, logger

DB_PATH = PROJECT_ROOT / "data" / "job_applications.db"

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------
def _connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = _connect()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS live_jobs (
            job_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            region       TEXT,
            title        TEXT,
            url          TEXT NOT NULL UNIQUE,
            snippet      TEXT,
            field        TEXT,
            email        TEXT,
            collected_at TEXT,
            applied      INTEGER NOT NULL DEFAULT 0,
            applied_at   TEXT
        )
        """
    )
    conn.commit()
    conn.close()


# Obfuscated emails commonly used on job boards to dodge scrapers:
#   name [at] domain [dot] com | name (at) domain (dot) com
#   name at domain dot com      | name@domain [dot] com
OBFUSCATED_RE = re.compile(
    r"([A-Za-z0-9._%+-]+)\s*(?:\[at\]|\(at\)|@|\bat\b)\s*"
    r"([A-Za-z0-9.-]+?)\s*(?:\[dot\]|\(dot\)|\.|\bdot\b)\s*"
    r"([A-Za-z]{2,24})",
    re.IGNORECASE,
)


def extract_email(text: str) -> str:
    """Best-effort extraction of the first email address in a string.

    Handles both plain addresses (a@b.com) and common anti-scraper
    obfuscations (a [at] b [dot] com, a(at)b(dot)com, a at b dot com).
    """
    if not text:
        return ""
    text = re.sub(r"\s+", " ", str(text))
    m = EMAIL_RE.search(text)
    if m:
        return m.group(0)
    m = OBFUSCATED_RE.search(text)
    if m:
        return f"{m.group(1).strip()}@{m.group(2).strip()}.{m.group(3).strip()}"
    return ""


def save_jobs(df: pd.DataFrame, field_col: str = "field") -> int:
    """
    Upsert jobs into the DB by unique URL.

    New jobs are inserted; jobs already present are left untouched so the
    `applied` state is never reset by re-collecting.
    """
    init_db()
    conn = _connect()
    inserted = 0
    now = datetime.now().isoformat()
    for _, row in df.iterrows():
        url = str(row.get("url", "")).strip()
        if not url:
            continue
        email = extract_email(f"{row.get('title', '')} {row.get('snippet', '')}")
        cur = conn.execute(
            "INSERT OR IGNORE INTO live_jobs "
            "(region, title, url, snippet, field, email, collected_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(row.get("region", "")),
                str(row.get("title", ""))[:300],
                url,
                str(row.get("snippet", ""))[:2000],
                str(row.get(field_col, "")),
                email,
                str(row.get("collected_at", now)),
            ),
        )
        inserted += cur.rowcount
    conn.commit()
    conn.close()
    logger.info("save_jobs: %d new jobs inserted into %s", inserted, DB_PATH)
    return inserted


def load_jobs(include_applied: bool = True) -> pd.DataFrame:
    init_db()
    conn = _connect()
    df = pd.read_sql_query("SELECT * FROM live_jobs ORDER BY job_id DESC", conn)
    conn.close()
    if not include_applied and not df.empty:
        df = df[df["applied"] == 0]
    return df


def mark_applied(urls, applied_at: str | None = None) -> int:
    """Mark the given job URLs as applied. Returns number of rows updated."""
    if not urls:
        return 0
    at = applied_at or datetime.now().isoformat()
    init_db()
    conn = _connect()
    n = 0
    for url in urls:
        cur = conn.execute(
            "UPDATE live_jobs SET applied = 1, applied_at = ? WHERE url = ?",
            (at, url),
        )
        n += cur.rowcount
    conn.commit()
    conn.close()
    logger.info("mark_applied: %d jobs marked as applied", n)
    return n


def get_applied_urls() -> set:
    init_db()
    conn = _connect()
    rows = conn.execute("SELECT url FROM live_jobs WHERE applied = 1").fetchall()
    conn.close()
    return {r["url"] for r in rows}


# ---------------------------------------------------------------------------
# PDFs
# ---------------------------------------------------------------------------
def list_pdfs(directory: str) -> list:
    """Return the paths of all .pdf files directly inside `directory`."""
    p = Path(directory)
    if not p.is_dir():
        return []
    return sorted(str(f) for f in p.glob("*.pdf") if f.is_file())


# ---------------------------------------------------------------------------
# Email
# ---------------------------------------------------------------------------
def get_smtp_config():
    return {
        "host": os.getenv("SMTP_HOST", ""),
        "port": int(os.getenv("SMTP_PORT", "587")),
        "user": os.getenv("SMTP_USER", ""),
        "password": os.getenv("SMTP_PASS", ""),
        "from_email": os.getenv("EMAIL_FROM", ""),
    }


def send_application(recipient: str, subject: str, body: str, attachments: list) -> tuple:
    """
    Send one application email (with PDF attachments) via SMTP.

    Returns (ok: bool, message: str).
    """
    cfg = get_smtp_config()
    if not (cfg["host"] and cfg["from_email"]):
        return False, "SMTP not configured (set SMTP_HOST / SMTP_PORT / SMTP_USER / SMTP_PASS / EMAIL_FROM in .env)"
    if cfg["user"] and not cfg["password"]:
        return False, "SMTP password missing — add your App Password to SMTP_PASS in .env"
    if not recipient or "@" not in recipient:
        return False, "No valid recipient email — add a contact email for this job"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg["from_email"]
    msg["To"] = recipient
    msg.set_content(body)

    for path in attachments:
        p = Path(path)
        if p.is_file():
            msg.add_attachment(
                p.read_bytes(), maintype="application", subtype="pdf", filename=p.name
            )

    try:
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=30) as server:
            server.starttls()
            if cfg["user"]:
                server.login(cfg["user"], cfg["password"])
            server.send_message(msg)
        logger.info("Application sent to %s (%d attachments)", recipient, len(attachments))
        return True, f"✅ Sent to {recipient}"
    except Exception as e:  # noqa: BLE001
        logger.error("SMTP error to %s: %s", recipient, e)
        return False, f"❌ Failed ({recipient}): {e}"


if __name__ == "__main__":
    init_db()
    print(f"Database ready: {DB_PATH}")
    print(f"Jobs stored: {len(load_jobs())}")
