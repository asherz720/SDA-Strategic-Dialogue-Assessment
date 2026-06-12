"""OPTIONAL reference: one way to run a model and get structured output.

You do NOT have to use this module. SDA only needs a table of parsed
annotations (see ``sda.schema``); how you produce it is up to you. This file
shows the setup we used in the paper — a local model served with vLLM behind an
OpenAI-compatible endpoint — so you have a concrete starting point.

Serve a model first, e.g.::

    pip install vllm
    vllm serve Qwen/Qwen2.5-32B-Instruct --port 8000

Then::

    from sda.inference import VLLMJury
    jury = VLLMJury(model="Qwen/Qwen2.5-32B-Instruct")
    df = jury.annotate_csv("full_collected_set/.../dialogue.csv",
                           history_col="all_history", answer_col="answer")
    df.to_csv("my_model_raw.csv", index=False)   # 'raw_response' column added

`annotate_csv` only fills a ``raw_response`` column with the model's text. You
parse that text into the schema columns yourself (any JSON extractor works),
then hand the parsed table to ``sda.validate`` and ``sda.metrics``.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Default prompt template shipped in the repo. Must contain {history} and {answer}.
DEFAULT_PROMPT = Path(__file__).resolve().parent.parent / "prompts" / "with_constitution.txt"


def load_prompt(path=DEFAULT_PROMPT) -> str:
    """Load a prompt template.

    The shipped ``prompts/*.txt`` are copy-pasted Python snippets: the actual
    prompt is the f-string body inside triple quotes, with ``{{``/``}}`` escaping
    literal JSON braces. This extracts that body. The result still contains the
    ``{history}`` and ``{answer}`` placeholders (filled in by :meth:`fill`).
    """
    text = Path(path).read_text()
    if '"""' in text:                       # extract the f-string body
        text = text.split('"""', 2)[1] if text.count('"""') >= 2 else text
    if "{history}" not in text or "{answer}" not in text:
        raise ValueError(f"Prompt template {path} must contain {{history}} and {{answer}} placeholders.")
    return text


def fill(template, history, answer) -> str:
    """Reproduce the original f-string: substitute history/answer, unescape {{ }}."""
    return (template
            .replace("{{", "\x00").replace("}}", "\x01")
            .replace("{history}", str(history))
            .replace("{answer}", str(answer))
            .replace("\x00", "{").replace("\x01", "}"))


class VLLMJury:
    """Thin wrapper over an OpenAI-compatible (vLLM) chat endpoint."""

    def __init__(self, model, *, base_url="http://localhost:8000/v1", api_key="EMPTY",
                 prompt_template=None, temperature=0.1):
        # Imported lazily so the rest of the package needs no openai install.
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.temperature = temperature
        self.prompt_template = prompt_template or load_prompt()

    def judge(self, history, answer) -> str:
        """Return the model's raw text response for one (history, answer) turn."""
        content = fill(self.prompt_template, history, answer)
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": content}],
        )
        return resp.choices[0].message.content.strip()

    def annotate_csv(self, file_path, *, history_col="all_history", answer_col="answer",
                     sleep=0.0, checkpoint=None):
        """Run the model over every row, returning the df with a ``raw_response`` column."""
        df = pd.read_csv(file_path)
        df["raw_response"] = None
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            try:
                df.at[idx, "raw_response"] = self.judge(row[history_col], row[answer_col])
            except Exception as e:  # keep going; record the failure in-line
                df.at[idx, "raw_response"] = f"ERROR: {e}"
            if sleep:
                time.sleep(sleep)
            if checkpoint and idx % 50 == 0:
                df.to_csv(checkpoint, index=False)
        return df
