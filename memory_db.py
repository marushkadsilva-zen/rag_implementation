import sqlite3

DB_NAME = "memory.db"


def init_memory_db():

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        key TEXT,
        value TEXT
    )
    """)

    conn.commit()
    conn.close()


def save_memory(key, value):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO user_memory (key, value) VALUES (?, ?)",
        (key, value)
    )

    conn.commit()
    conn.close()


def get_memory(key):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT value FROM user_memory WHERE key=? ORDER BY id DESC LIMIT 1",
        (key,)
    )

    result = cursor.fetchone()

    conn.close()

    if result:
        return result[0]

    return None