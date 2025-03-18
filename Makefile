default:
	@pytest
	
download_models:
	@ollama pull all-minilm:l6-v2
	@ollama pull initium/law_model

install_test_dependencies:
	@pip install types-setuptools types-reportlab types-PyYAML
	@pip install reportlab black isort flake8 mypy bandit pytest