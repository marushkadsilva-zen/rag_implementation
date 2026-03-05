import re
from memory_db import save_memory


def extract_memory(message):

    # Extract name
    name_pattern = r"my name is ([A-Za-z]+)"

    match = re.search(name_pattern, message.lower())

    if match:
        name = match.group(1)
        save_memory("user_name", name)
        print(f"Memory saved: user_name = {name}")