"""Check direct-choice protocol before evaluating all questions."""

import json
import re
import sys
import urllib.request
from pathlib import Path

port = int(sys.argv[1])
destination = Path(sys.argv[2])
rows = []
for prompt in [
    "What is 2+2? A) 3 B) 4 C) 5 D) 6",
    "Which element has atomic number 1? A) Helium B) Carbon C) Hydrogen D) Oxygen",
]:
    body = dict(
        model="glm53flash",
        messages=[
            dict(role="user", content=prompt + " Reply with only the letter."),
            dict(role="assistant", content="Answer:"),
        ],
        add_generation_prompt=False,
        continue_final_message=True,
        max_tokens=32,
        temperature=0,
        chat_template_kwargs=dict(reasoning_effort="low"),
    )
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        row = json.load(response)
    choice = row["choices"][0]
    content = choice["message"]["content"] or ""
    rows.append(row)
    assert choice["finish_reason"] != "length", row
    assert re.match(r"^\s*(?:Answer:\s*)?[ABCD]\b", content), row
destination.write_text(json.dumps(rows, indent=2))
