"""Minimal final-answer-only dataset input; no intermediate supervision."""
import json
from pathlib import Path


def load_final_answers(path):
    rows, seen = [], set()
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip(): continue
        item = json.loads(line)
        if any(not isinstance(item.get(k), str) or not item[k].strip() for k in ('id', 'question', 'answer')):
            raise ValueError(f'Line {number}: nonempty id/question/answer required')
        if item['id'] in seen: raise ValueError(f'Duplicate source ID at line {number}')
        seen.add(item['id'])
        rows.append({key: item[key] for key in ('id', 'question', 'answer')})
    if not rows: raise ValueError('Empty dataset')
    return rows
