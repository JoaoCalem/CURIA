start_app:
	@python curia/frontend/app.py

test:
	@pytest
	
download_models:
	@ollama pull all-minilm:l6-v2
	@ollama pull initium/law_model
	@ollama pull gemma3:1b

install_test_dependencies:
	@pip install types-setuptools types-reportlab types-PyYAML
	@pip install reportlab black isort flake8 mypy bandit pytest