# run_jailbreak_all_ollama.py

from data import JailbreakQueries
from attacks import Jailbreak
from metrics import JailbreakRate
from ollama_wrapper import OllamaModel
import os

# List of Ollama model names to test
OLLAMA_MODELS = ["llama2", "mistral", "gemma", "phi", "deepseek-r1"]

# Ensure results directory exists
os.makedirs("results", exist_ok=True)
csv_path = "results/jailbreak_ollama_results.csv"

# Start running
print("🚀 Running Jailbreak Attack across Ollama models...\n")

for model_name in OLLAMA_MODELS:
    print(f"🧠 Testing model: {model_name}")

    # Load dataset and model
    data = JailbreakQueries()
    llm = OllamaModel(model_name=model_name)

    # Run Jailbreak attack
    attack = Jailbreak()
    results = attack.execute_attack(data, llm)

    # Evaluate and print result
    rate = JailbreakRate(results).compute_metric()
    print(f"✅ Jailbreak Rate for {model_name}: {rate:.4f}\n")

    # Log to CSV
    with open(csv_path, "a") as f:
        f.write(f"{model_name},Jailbreak,{rate:.4f}\n")
