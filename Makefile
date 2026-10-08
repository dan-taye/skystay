# Thin wrapper around the `skystay` CLI (the CLI is the primary interface; it also works on Windows without make).
.PHONY: setup data train backtest sensitivity app test video deck all clean install

install:
	python -m venv .venv && .venv/bin/pip install -e . && .venv/bin/python -m playwright install chromium

setup data train backtest sensitivity app test video deck all clean:
	skystay $@
