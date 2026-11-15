from __future__ import annotations
import os
import sqlite3
import hashlib
from pathlib import Path
from cryptography.fernet import Fernet
import pandas as pd

# Supports a persistent mount path in production via AMRCE_DB_PATH env var
DB_PATH = Path(os.getenv("AMRCE_DB_PATH", "amrce_users.db"))
KEY_PATH = Path(os.getenv("AMRCE_KEY_PATH", "amrce_secret.key"))


def _load_fernet() -> Fernet:
    """
    Loads encryption key from AMRCE_ENCRYPTION_KEY env var (recommended for production deployment)
    or falls back to local amrce_secret.key file.
    """
    env_key = os.getenv("AMRCE_ENCRYPTION_KEY", "").strip()
    if env_key:
        return Fernet(env_key.encode("utf-8"))
    if KEY_PATH.exists():
        return Fernet(KEY_PATH.read_bytes())
    key = Fernet.generate_key()
    KEY_PATH.write_bytes(key)
    return Fernet(key)


FERNET = _load_fernet()


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def encrypt_secret(plain_text: str) -> str:
    if not plain_text:
        return ""
    return FERNET.encrypt(plain_text.strip().encode("utf-8")).decode("utf-8")


def decrypt_secret(enc_text: str) -> str:
    if not enc_text:
        return ""
    try:
        return FERNET.decrypt(enc_text.encode("utf-8")).decode("utf-8")
    except Exception:
        return ""


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        # 1. Users & Encrypted Broker Vault Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                is_approved INTEGER NOT NULL DEFAULT 0,
                enc_api_key TEXT DEFAULT '',
                enc_secret_key TEXT DEFAULT '',
                paper_mode INTEGER NOT NULL DEFAULT 1,
                active_split INTEGER NOT NULL DEFAULT 60,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        """)

        # 2. Per-User Executed Trade Ledger Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                notional_usd REAL NOT NULL,
                account_equity REAL DEFAULT 0.0,
                environment TEXT NOT NULL DEFAULT 'PAPER',
                status TEXT NOT NULL,
                alpaca_order_id TEXT DEFAULT '',
                executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)

        # 3. Security & Admin Audit Log Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                actor TEXT NOT NULL,
                action TEXT NOT NULL,
                details TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Ensure default founder admin exists
        cur.execute("SELECT COUNT(*) FROM users WHERE role = 'admin'")
        if cur.fetchone()[0] == 0:
            cur.execute(
                """
                INSERT INTO users (username, password_hash, role, is_approved)
                VALUES (?, ?, 'admin', 1)
                """,
                ("admin", hash_password("adminpassword123")),
            )
        conn.commit()


init_db()


def log_audit_event(actor: str, action: str, details: str = "") -> None:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO audit_logs (actor, action, details) VALUES (?, ?, ?)",
            (actor, action, details),
        )
        conn.commit()


def authenticate_user(username: str, password: str) -> dict | None:
    u = username.strip()
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, username, role, is_approved, enc_api_key, enc_secret_key, paper_mode
            FROM users
            WHERE username = ? AND password_hash = ?
            """,
            (u, hash_password(password)),
        )
        row = cur.fetchone()
        if row:
            cur.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (row[0],))
            conn.commit()

    if not row:
        return None

    log_audit_event(row[1], "USER_LOGIN", f"Role: {row[2]}")
    return {
        "id": row[0],
        "username": row[1],
        "role": row[2],
        "is_approved": bool(row[3]),
        "api_key": decrypt_secret(row[4]),
        "secret_key": decrypt_secret(row[5]),
        "paper_mode": bool(row[6]),
    }


def register_user(username: str, password: str) -> tuple[bool, str]:
    u = username.strip()
    if len(u) < 3 or len(password) < 6:
        return False, "Username must be at least 3 characters and password at least 6 characters."
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO users (username, password_hash, role, is_approved)
                VALUES (?, ?, 'user', 0)
                """,
                (u, hash_password(password)),
            )
            conn.commit()
        log_audit_event(u, "ACCOUNT_REQUESTED", "Pending admin approval")
        return True, "Account requested. A founder/admin must approve your access before sign-in."
    except sqlite3.IntegrityError:
        return False, "Username already exists."


def create_user_by_admin(actor: str, username: str, password: str, role: str = "user", approved: bool = True) -> tuple[bool, str]:
    u = username.strip()
    if len(u) < 3 or len(password) < 6:
        return False, "Username >= 3 chars and password >= 6 chars required."
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO users (username, password_hash, role, is_approved)
                VALUES (?, ?, ?, ?)
                """,
                (u, hash_password(password), role, int(approved)),
            )
            conn.commit()
        log_audit_event(actor, "CREATE_USER", f"Created {u} with role={role}")
        return True, f"Created account '{u}' ({role.upper()})."
    except sqlite3.IntegrityError:
        return False, "Username already exists."


def update_user_credentials(username: str, api_key: str, secret_key: str, paper_mode: bool) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE users
            SET enc_api_key = ?, enc_secret_key = ?, paper_mode = ?
            WHERE username = ?
            """,
            (encrypt_secret(api_key), encrypt_secret(secret_key), int(paper_mode), username),
        )
        conn.commit()
    log_audit_event(username, "VAULT_UPDATED", f"Mode={'PAPER' if paper_mode else 'LIVE'}")


def update_user_password(actor: str, target_username: str, new_password: str) -> tuple[bool, str]:
    if len(new_password) < 6:
        return False, "Password must be at least 6 characters."
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET password_hash = ? WHERE username = ?",
            (hash_password(new_password), target_username),
        )
        conn.commit()
    log_audit_event(actor, "PASSWORD_UPDATED", f"Target user: {target_username}")
    return True, f"Password updated for '{target_username}'."


def record_user_trade(
    user_id: int,
    username: str,
    symbol: str,
    side: str,
    notional_usd: float,
    account_equity: float,
    paper_mode: bool,
    status: str,
    alpaca_order_id: str = "",
) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO user_trades (
                user_id, username, symbol, side, notional_usd,
                account_equity, environment, status, alpaca_order_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(user_id),
                username,
                symbol,
                side,
                float(notional_usd),
                float(account_equity),
                "PAPER" if paper_mode else "LIVE",
                status,
                str(alpaca_order_id),
            ),
        )
        conn.commit()


def get_user_recorded_trades(user_id: int | None = None) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        if user_id is None:
            return pd.read_sql_query(
                """
                SELECT executed_at AS "Timestamp", username AS "User", symbol AS "Symbol",
                       side AS "Side", notional_usd AS "Order Notional ($)",
                       account_equity AS "Account Equity ($)", environment AS "Mode",
                       status AS "Status", alpaca_order_id AS "Order ID"
                FROM user_trades
                ORDER BY id DESC LIMIT 500
                """,
                conn,
            )
        return pd.read_sql_query(
            """
            SELECT executed_at AS "Timestamp", symbol AS "Symbol", side AS "Side",
                   notional_usd AS "Order Notional ($)", account_equity AS "Account Equity ($)",
                   environment AS "Mode", status AS "Status", alpaca_order_id AS "Order ID"
            FROM user_trades
            WHERE user_id = ?
            ORDER BY id DESC LIMIT 250
            """,
            conn,
            params=(int(user_id),),
        )


def get_all_users() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            """
            SELECT u.id, u.username, u.role, u.is_approved, u.paper_mode,
                   CASE WHEN u.enc_api_key != '' THEN 1 ELSE 0 END AS has_keys,
                   u.created_at, u.last_login,
                   COUNT(t.id) AS total_trades,
                   COALESCE(SUM(t.notional_usd), 0.0) AS total_volume
            FROM users u
            LEFT JOIN user_trades t ON u.id = t.user_id
            GROUP BY u.id
            ORDER BY u.id ASC
            """,
            conn,
        )


def set_user_approval(actor: str, user_id: int, approved: int) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("UPDATE users SET is_approved = ? WHERE id = ?", (int(approved), int(user_id)))
        conn.commit()
    log_audit_event(actor, "SET_APPROVAL", f"user_id={user_id} approved={approved}")


def set_user_role(actor: str, user_id: int, role: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("UPDATE users SET role = ?, is_approved = 1 WHERE id = ?", (role, int(user_id)))
        conn.commit()
    log_audit_event(actor, "SET_ROLE", f"user_id={user_id} role={role}")


def delete_user_account(actor: str, user_id: int) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM user_trades WHERE user_id = ?", (int(user_id),))
        cur.execute("DELETE FROM users WHERE id = ?", (int(user_id),))
        conn.commit()
    log_audit_event(actor, "DELETE_USER", f"user_id={user_id}")


def get_audit_logs(limit: int = 150) -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        return pd.read_sql_query(
            """
            SELECT created_at AS "Timestamp", actor AS "Actor",
                   action AS "Event", details AS "Details"
            FROM audit_logs
            ORDER BY id DESC LIMIT ?
            """,
            conn,
            params=(int(limit),),
        )