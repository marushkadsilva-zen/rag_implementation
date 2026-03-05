import sqlite3
from datetime import datetime

DB_NAME = "multi_doc_history.db"


def init_multi_doc_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS multi_doc_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id TEXT,
        role TEXT,
        message TEXT,
        sources TEXT,
        timestamp TEXT
    )
    """)

    conn.commit()
    conn.close()

def save_multi_doc_message(chat_id, role, message, sources=None):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO multi_doc_history
        (chat_id, role, message, sources, timestamp)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            chat_id,
            role,
            message,
            sources,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    )

    conn.commit()
    conn.close() 
def get_multi_doc_history(chat_id):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT role, message
        FROM multi_doc_history
        WHERE chat_id=?
        ORDER BY id
        """,
        (chat_id,)
    )

    rows = cursor.fetchall()

    conn.close()

    return rows
def clear_multi_doc_chat(chat_id):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM multi_doc_history WHERE chat_id=?",
        (chat_id,)
    )

    conn.commit()
    conn.close()