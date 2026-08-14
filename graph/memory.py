# graph/memory.py — SQLite checkpointer setup (reuses existing newchatbot.db)

import sqlite3
# pyrefly: ignore [missing-import]
from langgraph.checkpoint.sqlite import SqliteSaver

_conn = None
_checkpointer = None


def get_checkpointer() -> SqliteSaver:
    """Return a singleton SqliteSaver using the project's existing database."""
    global _conn, _checkpointer
    if _checkpointer is None:
        _conn = sqlite3.connect("newchatbot.db", check_same_thread=False)
        _checkpointer = SqliteSaver(_conn)
    return _checkpointer


def get_connection():
    """Return the raw sqlite3 connection (used for thread deletion in the UI)."""
    get_checkpointer()  # ensure init
    return _conn
