"""One trace per question, appended to a JSONL file.

A trace records what was retrieved, which prompt versions and model were used,
how long each step took and what came out. It is enough to answer "why did the
assistant say that?" after the fact, and it is what the eval script reads.

Note for anything beyond a prototype: traces contain the question and the
answer in clear text, so with real clinical data they need the same access
control and retention rules as the documents themselves.
"""

import json
import threading
from pathlib import Path


class TraceStore:
    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, trace: dict) -> None:
        line = json.dumps(trace, ensure_ascii=False, default=str)
        with self._lock, self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def get(self, trace_id: str) -> dict | None:
        for trace in self._read():
            if trace["trace_id"] == trace_id:
                return trace
        return None

    def recent(self, limit: int = 20) -> list[dict]:
        return self._read()[-limit:][::-1]

    def _read(self) -> list[dict]:
        # Linear scan. Fine for a local prototype; a real deployment would send
        # these to a tracing backend instead of reading a file.
        if not self._path.exists():
            return []
        with self._path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
