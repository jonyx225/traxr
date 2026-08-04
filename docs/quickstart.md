# Quickstart

## Bring your agent

Your agent is any callable `(Task) -> str`. Wrap its OpenAI client with
[`traxr.instrument()`](api.md#capture) and every
`chat.completions` call (sync, async, or streaming, including tool calls)
is captured into the trace:

```python
import openai, traxr

# Wrap your OpenAI client once so each call is recorded as a trace step.
client = traxr.instrument(openai.OpenAI())

def my_agent(task: traxr.Task) -> str:
    messages = [{"role": "user", "content": task.question}]
    while True:
        reply = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=tools
        ).choices[0].message
        messages.append(reply)
        if not reply.tool_calls:                 # no tool call -> final answer
            return reply.content or ""
        for call in reply.tool_calls:
            output = run_tool(call, task.files)   # your tool reads the data
            messages.append({"role": "tool",
                             "tool_call_id": call.id, "content": output})

experiment = traxr.Experiment(
    files="examples/sales.csv",
    question="Which region had the highest Q3 revenue?",
    expected_answer="EMEA",
    agent=my_agent,
)
experiment.run(dry_run=True)   # the full plan: zero LLM calls, zero spend
results = experiment.run()     # baseline + perturbed runs (+ noise floor)
print(results.summary())
results.to_json("results.json")
```

Here `tools` is your function-schema list and `run_tool` runs the selected
tool over `task.files`. Stateful agents (memory, vector stores) should pass
`agent_factory=` (a zero-arg factory called once per run) so every run starts
fresh.

## Scoring free-text answers

The default scorer, `check_answer_match`, is normalized string equality
with numeric tolerance, exact on purpose. A real agent answers in full
sentences ("The region with the highest Q3 revenue is EMEA, with a
revenue of 181,400."), which will never literally equal a bare
`expected_answer="EMEA"`. Bring your own scorer via `ExperimentConfig`:

```python
from traxr import ExperimentConfig
from traxr.scoring import llm_judge_match

experiment = traxr.Experiment(
    files="examples/sales.csv",
    question="Which region had the highest Q3 revenue?",
    expected_answer="EMEA",
    agent=my_agent,
    config=ExperimentConfig(scorer=llm_judge_match),
)
```

`llm_judge_match` asks an LLM whether the candidate answer reaches the
same core conclusion as the expected one. Useful for verbose answers,
but **not deterministic**: it costs an extra LLM call per scored answer
and can vary between runs. By default it lazily builds an
`OpenAICompatibleClient(model="gpt-4o-mini")` from `OPENAI_API_KEY`; pass
`functools.partial(llm_judge_match, llm=my_llm)` to use another provider.
A `scorer` is just `(expected, actual) -> bool`, so a plain function works
too if you want something deterministic but less strict than the default.

## Expose your agents with `traxr.emit`

`instrument()` captures your LLM traffic automatically, but it cannot see
*which* agent is acting; those calls are tagged `agent_name="external"`. In a
multi-agent system that's the part you most want in the trace. [`traxr.emit()`](api.md#capture) is the escape
hatch: call it from inside your code, where you know who is in charge and what
they just decided.

```python
# agent_name = the agent acting now; chosen_agent = who the orchestrator picks next.
traxr.emit("routing_decision", {"chosen_agent": "researcher"}, agent_name="orchestrator")
```

Two rules carry most of the value:

- **Actor vs. destination.** `agent_name` is *who is acting now*; the payload's
  `chosen_agent` is *who runs next*. Read the `chosen_agent`s down the trace and
  you reconstruct the route, which is exactly what lets the metrics flag a
  *reroute* when a perturbation changes it.
- **Emit at decision points** the automatic capture misses: route changes,
  handoffs, memory reads, retrieval. Keep `instrument()` for the LLM/tool
  traffic; don't duplicate it with `emit`.

Outside a Traxr run `emit()` is a **no-op** (same passthrough principle as
`instrument()`), so the calls are safe to leave in production code.

### Centralized orchestrator

If one orchestrator decides who runs next, the routing event belongs to *it*,
not to the workers. Emit one `routing_decision` per hop from the orchestrator;
each worker emits its own `agent_output` / tool events:

```python
def orchestrate(task: traxr.Task) -> str:
    state = init_state(task)
    while not done(state):
        next_agent = decide_next_agent(state)        # your routing logic
        traxr.emit(
            "routing_decision",
            {"chosen_agent": next_agent},             # destination = who you picked
            agent_name="orchestrator",                # actor = the orchestrator
        )
        state = run_agent(next_agent, state)          # worker emits its own events
    return final_answer(state)
```

### Custom event types

Reusing the built-in types (`routing_decision`, `agent_output`,
`tool_invocation`, `memory_write`) makes events count toward `d_norm` out of
the box. A custom type still appears in the trace JSON, but its signature
collapses to `unknown:<type>` until you upgrade it with
[`register_signature()`](api.md#extending-the-event-vocabulary):

```python
# once, before experiment.run()
traxr.register_signature(
    "handoff",
    lambda p: f"handoff:{p.get('from_agent')}->{p.get('to_agent')}",
    classifier=lambda clean, pert: (
        "different_handoff" if clean.get("to_agent") != pert.get("to_agent") else None
    ),
    structural_types={"different_handoff"},
)
```

## The CLI

The same experiment from the shell. `--agent` takes an `import:path` to your
callable:

```bash
traxr run --agent mypkg.agents:answer --file examples/sales.csv \
          --question "Which region won Q3?" --expected-answer EMEA \
          --out results.json
traxr run --model gpt-4o-mini --file report.pdf --question "..." --dry-run
traxr operators
traxr selfcheck
```

Two spend knobs: `--max-llm-calls N` caps LLM calls per run for an external
`--agent` (the `max_llm_calls_per_run` budget, default 50; `RunBudgetExceeded`
fires before the over-budget call), and `--max-retries N` sets how many times
the built-in `--model` client retries a transient failure (default 2, the
OpenAI SDK default; `0` disables). In Python they are
`ExperimentConfig(max_llm_calls_per_run=...)` and
`OpenAICompatibleClient(max_retries=...)`.

## Use the built-in reference system

No agent of your own? Traxr ships a multi-agent reference system (planner,
researcher, tools, synthesizer). Point it at your data with `llm=` and an
API key:

```python
import traxr

experiment = traxr.Experiment(
    files="examples/sales.csv",
    question="Which region had the highest Q3 revenue?",
    expected_answer="EMEA",
    llm=traxr.OpenAICompatibleClient(model="gpt-4o-mini"),  # reads OPENAI_API_KEY
)
results = experiment.run()
```

Swap the stub for a real endpoint with
[`OpenAICompatibleClient`](api.md#llm-clients) — OpenAI, Azure, Ollama,
vLLM, Together, Groq, OpenRouter, anything OpenAI-compatible.

## Try it on a real benchmark task

[`examples/gaia_menu_sales.py`](https://github.com/anna-mazhar/traxr/blob/main/examples/gaia_menu_sales.py)
runs a real GAIA benchmark question end-to-end through Traxr's built-in
multi-agent system — no setup required (falls back to a deterministic stub
without an API key, upgrades automatically to a real `gpt-4o-mini` agent if
`OPENAI_API_KEY` is set).

### Sample output

Run it with no API key and `print(results.summary())` gives the compact
diagnosis — how the perturbations manifested, how far the traces diverged,
whether the answer survived, and what it cost:

```text
Traxr experiment summary
  agent kind: builtin (capture: builtin)
  pairs: 7 (7 scored, 0 skipped)
  manifestations:
     71%  loop_or_extended_execution
     29%  no_observable_effect
  d_norm: mean 0.3702, max 0.7857 over 7 pair(s)
  noise floor (d_norm): unmeasured (deterministic built-in path)
  recovery rate: 40% of diverged pairs kept their answer
  token overhead: mean 2.53x
```

With `OPENAI_API_KEY` set (`gpt-4o-mini`, `scorer=llm_judge_match`), the
per-pair report (`results.to_report()`) is where the trace-level signal
shows up — one real run produced:

| perturbation | manifestation | answer changed | task succeeded | `d_norm` | `t*` | token overhead |
|---|---|---|---|---|---|---|
| `column_swap` | no_observable_effect | no | yes | 0.00 | — | 0.99x |
| `label_corrupt` | loop_or_extended_execution | yes | no | 0.69 | 0 | 3.02x |
| `data_type_corrupt` | loop_or_extended_execution | no | yes | 0.56 | 0 | 4.46x |
| `row_duplicate` | loop_or_extended_execution | yes | no | 0.25 | 2 | 1.56x |
| `irrelevant_columns` | loop_or_extended_execution | no | yes | 0.25 | 2 | 1.45x |
| `unit_change` | silent_semantic_corruption | yes | no | 0.00 | — | 1.00x |
| `null_content` | catastrophic_failure | yes | — | 0.65 | 2 | — |

Two rows (`data_type_corrupt`, `irrelevant_columns`) still landed the right
answer — but only after the trace visibly lengthened (token overhead up to
4.46x), which an answer-only harness would score as clean passes.
`unit_change` is the inverse: the answer flipped while the trace looked
structurally identical (`d_norm=0.0`). See
[examples/README.md](https://github.com/anna-mazhar/traxr/blob/main/examples/README.md)
for the full writeup. Your own numbers will differ (LLM nondeterminism,
model version) — rerun and compare against this table rather than treating
it as fixed.

## LangGraph

```python
agent = traxr.from_langgraph(compiled_graph)
experiment = traxr.Experiment(files="report.pdf", question="...", agent=agent)
```

Node transitions become routing events (and carry agent names onto LLM events
automatically), tool calls keep success/failure fidelity, and double-counting
with an instrumented client is suppressed. For non-messages-state graphs, pass
`input_builder=` / `output_extractor=`.

## The notebook

[`notebooks/traxr_quickstart.ipynb`](https://github.com/anna-mazhar/traxr/blob/main/notebooks/traxr_quickstart.ipynb)
runs top-to-bottom without an API key. Open it in Colab.
