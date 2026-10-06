# Sourced by check.sh and demo.sh. Interpreter: $PYTHON if set, else $PORTFOLIO_VENV/bin/python, else ./.venv/bin/python.
# Without one, say how to make it and stop: a missing environment must not read as a pass.
if [ -n "${PYTHON:-}" ]; then PY="$PYTHON"
elif [ -n "${PORTFOLIO_VENV:-}" ]; then PY="$PORTFOLIO_VENV/bin/python"
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
else
  echo "error: no Python environment. Create one: python3.12 -m venv .venv && .venv/bin/pip install . --group dev --group qiskit mpmath" >&2
  exit 3
fi
if ! "$PY" -c "import numpy, numba, mpmath, pytest" 2>/dev/null; then
  echo "error: $PY lacks the test packages: $PY -m pip install . --group dev --group qiskit mpmath" >&2
  exit 3
fi
# The checked-out source, not an installed copy, is what gets tested.
export PYTHONPATH="src"
