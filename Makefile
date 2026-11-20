PYTHON ?= python3

.PHONY: verify test run

verify:
	$(PYTHON) -m src.platform_data

test:
	$(PYTHON) -m pytest -q

run:
	$(PYTHON) -m streamlit run src/dashboard.py --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
