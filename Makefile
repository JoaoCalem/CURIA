default:
	@pytest
	
download_models:
	@ollama pull all-minilm:l6-v2
	@ollama pull initium/law_model