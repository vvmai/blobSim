# Task Completion Checklist

After completing any code change:

1. **Type check**: `pyright` — must pass cleanly
2. **Lint**: `ruff check .` — fix any issues
3. **Format**: `ruff format .` — ensure consistent style
4. **Tests**: `python -m pytest tests/ -v` — all tests must pass
5. **Conservation**: If touching engine/ledger/energy code, verify `python main.py` runs without conservation violation
6. **Benchmarks**: If touching core simulation logic, run `python -m benchmark.run_all` and compare results
