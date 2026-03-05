import sqlite3

DB_NAME = "chat_history.db"


def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id TEXT,
        role TEXT,
        message TEXT
    )
    """)

    conn.commit()
    conn.close()


def save_message(chat_id, role, message):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO chat_messages (chat_id, role, message) VALUES (?, ?, ?)",
        (chat_id, role, message)
    )

    conn.commit()
    conn.close()


def get_chat_history(chat_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT role, message FROM chat_messages WHERE chat_id=?",
        (chat_id,)
    )

    rows = cursor.fetchall()
    conn.close()

    return rows


def clear_chat(chat_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM chat_messages WHERE chat_id=?",
        (chat_id,)
    )

    conn.commit()
    conn.close()