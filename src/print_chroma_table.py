import os
import chromadb
from tabulate import tabulate  # pip install tabulate
from wcwidth import wcswidth

CHROMA_DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db"))
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_collection("articles")

results = collection.get(limit=30, include=['documents', 'metadatas'])

def short_kor(text, length):
    acc = ''
    acc_len = 0
    for ch in text:
        acc_len += wcswidth(ch)
        if acc_len > length:
            acc += '...'
            break
        acc += ch
    return acc

table = []
for i in range(len(results['ids'])):
    meta = results['metadatas'][i]
    row = [
        str(i+1),
        short_kor(meta.get('title', ''), 18),
        short_kor(meta.get('url', ''), 18),
        short_kor(meta.get('date', ''), 10),
        short_kor(meta.get('author', ''), 6),
        short_kor(meta.get('category', ''), 6),
        short_kor(results['documents'][i], 30)
    ]
    table.append(row)

headers = ["id", "title", "url", "date", "author", "category", "본문 일부"]
print(tabulate(table, headers=headers, tablefmt="fancy_grid", disable_numparse=True))