import os
import chromadb
from wcwidth import wcswidth
from rich.table import Table
from rich.console import Console

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

table = Table(show_header=True, header_style="bold magenta")
table.add_column("id", style="dim", width=4)
table.add_column("title", width=20)
table.add_column("url", width=20)
table.add_column("date", width=12)
table.add_column("author", width=8)
table.add_column("category", width=8)
table.add_column("본문 일부", width=32)

for i in range(len(results['ids'])):
    meta = results['metadatas'][i]
    table.add_row(
        str(i+1),
        short_kor(meta.get('title', ''), 18),
        short_kor(meta.get('url', ''), 18),
        short_kor(meta.get('date', ''), 10),
        short_kor(meta.get('author', ''), 6),
        short_kor(meta.get('category', ''), 6),
        short_kor(results['documents'][i], 30)
    )

console = Console()
console.print(table)