# deixis

**Tell any robot where.** `deixis` turns words like "here", "that one" and "this big", plus
pointing, into a place a robot can act on: a point, a box or a path in the robot's own frame,
with an uncertainty and the evidence, or one question when it is not sure.

It is pure code: no AI models, no camera, no internet, no robot. Same input, same output.
It depends only on Pydantic and NumPy.

It drives three robots, all over MCP: a pen plotter and a robot vacuum
([deixis-doodle](https://github.com/rahuljeyaraj/deixis-doodle)), and a virtual SO-101 arm in
the browser ([deixis-web-demo](https://github.com/rahuljeyaraj/deixis-web-demo)). All of them
are in one clone at [deixis-robots](https://github.com/rahuljeyaraj/deixis-robots).

What it cannot do is in [LIMITATIONS.md](LIMITATIONS.md). What changed is in
[CHANGELOG.md](CHANGELOG.md).

## How it works

The robot's app sends one request per sentence:

- the **words**, each with the time it was said;
- the **pointing**: fingertip, feet or mouse samples, or holds (where the pointer stayed still);
- the **things** in the scene (blocks, drawings), if any;
- the **tool calls** from the language model. A call never holds a coordinate. It points at
  a word instead: "here" is word 2.

`deixis` matches each pointing word to the hold made at that moment, and gives back a
`Where` (a place or a thing), a `Size`, or an `Ask` (one question, when two places fit).
A place outside the robot's work area is flagged, never moved.

Everything robot-specific comes from the robot's **profile**: its frame, its work area, its
actions, and whether it wants a point, a box or a path. See `examples/profiles/`.

## Set-up

You need [uv](https://docs.astral.sh/uv/) and Python 3.11 or newer. `deixis` is not on PyPI
yet, so install it from this repo:

```powershell
git clone https://github.com/rahuljeyaraj/deixis.git
cd deixis
uv sync
uv run pytest
```

From another project: `uv add "deixis[mcp] @ git+https://github.com/rahuljeyaraj/deixis.git"`
(the `mcp` extra is only needed for the MCP server and client).

## Use it from Python

"Put that here", with a fingertip on the red block at "that" and on an empty spot at "here":

```python
import json
from deixis import Request, load_profile, resolve

example = json.load(open("examples/requests/E4-arm-put-that-here.json"))
profile = load_profile("examples/profiles/arm-table.json")
response = resolve(Request.model_validate(example["request"]), profile)

move = response.calls[0].args
print(move["object"].thing)    # red
print(move["place"].mean_mm)   # (480.0, 90.0), in the table's frame
```

`examples/requests/` has seven requests like this one (plotter, vacuum and arm), including
one where `deixis` is not sure and asks.

## Use it over MCP

```powershell
uv run deixis-mcp --profile examples/profiles/arm-table.json
```

This starts an MCP server on stdio. Its tools: `describe`, `set_profile`, `resolve`, `adjust`
("a bit to the left"), `track`, `fit_transform`, `apply_transform`, `tools_for_agent` (the
tool list to give the language model) and `schema`. `--record DIR` logs every call.

## Messages

Every message has a JSON Schema in `schema/` (`where-1.json`, `request-1.json` and so on), so
an app in any language can talk to `deixis`. `uv run python -m deixis schema` writes them
again.

## Licence

MIT, see [LICENSE](LICENSE).
