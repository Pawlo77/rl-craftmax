"""Execute a Jupytext percent-format notebook (.py) top to bottom in a fresh kernel.

usage (self-contained; no project environment needed):
  uv run --no-project --python 3.12 \
    --with jupytext --with ipykernel --with jupyter-client --with nbformat \
    --with numpy --with matplotlib --with scipy --with pandas \
    python -I run_notebook.py <notebook.py> --out <notebook.ipynb> [--print] [--timeout 600]

Writes the executed notebook to --out (default: next to the .py). The delivered notebooks in
research/<book>/notebooks/ are .ipynb only: author the percent .py in a scratch directory and
point --out there. To edit a delivered notebook, convert it back with
`uv run --with jupytext jupytext --to py:percent --update-metadata '{"jupytext": {"formats": null}}' <file>.ipynb --output <scratch>.py`
(the notebooks record a py:percent pairing that jupytext otherwise enforces).
Prints one line per cell. With --print it also prints each cell's text output so the
markdown can be checked against it. Stops at the first failing cell and exits non-zero.
Plots are captured as image outputs (inline backend); set nothing in the notebook for that.
"""

import argparse
import json
import queue
import sys
import time
from pathlib import Path

import jupytext
import nbformat
from jupyter_client.manager import start_new_kernel


def lines(text: str) -> list[str]:
    parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])


def run(kc, code: str, timeout: int):
    msg_id = kc.execute(code, store_history=True)
    outputs, count, failed = [], None, False
    while True:
        try:
            msg = kc.get_iopub_msg(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"cell exceeded {timeout}s") from None
        if msg["parent_header"].get("msg_id") != msg_id:
            continue
        kind, c = msg["msg_type"], msg["content"]
        if kind == "status" and c["execution_state"] == "idle":
            break
        if kind == "execute_input":
            count = c["execution_count"]
        elif kind == "stream":
            if (
                outputs
                and outputs[-1]["output_type"] == "stream"
                and outputs[-1]["name"] == c["name"]
            ):
                outputs[-1]["text"] = lines("".join(outputs[-1]["text"]) + c["text"])
            else:
                outputs.append({
                    "name": c["name"],
                    "output_type": "stream",
                    "text": lines(c["text"]),
                })
        elif kind in ("execute_result", "display_data"):
            data = {
                k: (
                    lines(v) if isinstance(v, str) and not k.startswith("image/") else v
                )
                for k, v in c["data"].items()
            }
            out = {"data": data, "metadata": c.get("metadata", {}), "output_type": kind}
            if kind == "execute_result":
                out["execution_count"] = c["execution_count"]
            outputs.append(out)
        elif kind == "error":
            failed = True
            outputs.append({
                "ename": c["ename"],
                "evalue": c["evalue"],
                "output_type": "error",
                "traceback": c["traceback"],
            })
        elif kind == "clear_output":
            outputs = []
    return outputs, count, failed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--print", action="store_true", dest="show")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--no-ipynb", action="store_true")
    a = ap.parse_args()

    nb = json.loads(nbformat.writes(jupytext.read(a.notebook)))
    km, kc = start_new_kernel(
        kernel_name="python3", cwd=str(a.notebook.resolve().parent)
    )
    status, t0 = 0, time.time()
    try:
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            t = time.time()
            outputs, count, failed = run(kc, "".join(cell["source"]), a.timeout)
            cell["outputs"], cell["execution_count"] = outputs, count
            for o in outputs:
                if o["output_type"] == "execute_result":
                    o["execution_count"] = count
            print(
                f"cell {i}: {'ERROR' if failed else 'ok'} ({time.time() - t:.1f}s, {len(outputs)} outputs)"
            )
            if a.show:
                for o in outputs:
                    if o["output_type"] == "stream":
                        print("".join(o["text"]).rstrip())
                    elif (
                        o["output_type"] in ("execute_result", "display_data")
                        and "text/plain" in o["data"]
                    ):
                        print("".join(o["data"]["text/plain"]).rstrip())
            if failed:
                err = next(o for o in outputs if o["output_type"] == "error")
                print(f"  {err['ename']}: {err['evalue']}")
                status = 1
                break
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)
        if not a.no_ipynb:
            out = a.out or a.notebook.with_suffix(".ipynb")
            out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
            print(f"wrote {out} (total {time.time() - t0:.1f}s)")
    sys.exit(status)


main()
