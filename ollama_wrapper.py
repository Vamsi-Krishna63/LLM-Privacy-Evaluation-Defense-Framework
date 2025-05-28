 # ollama_wrapper.py

import requests

class OllamaModel:
    def __init__(self, model_name="llama2", api_url="http://localhost:11434/api/chat"):
        self.model_name = model_name
        self.api_url = api_url

    def query(self, prompt):
        try:
            response = requests.post(self.api_url, json={
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            })
            return response.json()["message"]["content"]
        except Exception as e:
            print("❌ Ollama query failed:", e)
            return f"[Ollama Error] {str(e)}"
