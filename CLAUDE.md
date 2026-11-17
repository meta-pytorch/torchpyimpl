# torchpyimpl

## Tests

- Multi-rank test workers use `@dist_worker` from
  `tests/distributed/symmetric_memory/_dist_worker.py`. It initializes the
  process group, runs the worker and destroys the group, so a worker is
  `fn(rank, world_size, *args)` with no setup code and no `try/finally` of its
  own. Spawn it with `mp.spawn(worker, args=(world_size, port, *args), ...)`.
- Do not wrap test bodies in `try/except` or `try/finally`. An exception must
  fail the test.
- Never catch `Exception` (or use a bare `except`) in tests or benchmarks;
  ruff enforces this with BLE001, S110 and S112. To check that a call raises,
  catch the specific exception type. For work on other threads, use
  `concurrent.futures.ThreadPoolExecutor` and call `future.result()`, which
  re-raises the thread's exception.
- A `try/finally` is fine only to undo process-wide state within the pytest
  process (environment variables, monkeypatched functions, driver
  allocations); prefer pytest's `monkeypatch` where it fits.

## Library code

- `torchpyimpl/` is exempt from BLE001. Its broad catches are deliberate:
  ranks hold a local error until every rank agrees whether setup failed, so
  that one rank never raises while its peers wait in a collective; cleanup
  paths log the traceback and report a failed status instead of raising.
  New broad catches must follow one of these two patterns.

## Style

- No `#` comments and no docstrings (module, class or function) in the
  symmetric memory code, tests and benchmarks. License headers and tool
  directives (`# type: ignore[...]`, `# pyrefly: ignore`, `# noqa`) stay.
  Put explanations in commit messages or `docs/`.

## Writing

- Write documentation (`docs/`, `README.md`), commit messages, PR
  descriptions, and error and log messages in ASD-STE100 Simplified
  Technical English.
- Use approved words with their approved meaning. You can use technical
  names, for example API names and CUDA terms.
- Write procedural sentences of 20 words or fewer and descriptive sentences
  of 25 words or fewer.
- Write one instruction in each sentence. Use the imperative for
  instructions.
- Use the active voice.
- Use simple verb tenses: present, past and future. Do not use the "-ing"
  form as a verb.
- Write paragraphs of six sentences or fewer. Each paragraph has one topic.
- Do not remove articles ("the", "a") or verbs to make text shorter.
