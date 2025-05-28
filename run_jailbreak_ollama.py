# run_jailbreak_ollama.py

from data import JailbreakQueries
from attacks import Jailbreak
from metrics import JailbreakRate
from ollama_wrapper import OllamaModel

if __name__ == "__main__":
    print("🚀 Starting Jailbreak Attack with Ollama...")

    # Step 1: Load dataset
    data = JailbreakQueries()

    # Step 2: Initialize model
    model_name = "llama2"  # You can change to: mistral, phi, gemma
    llm = OllamaModel(model_name=model_name)

    # Step 3: Run attack
    attack = Jailbreak()
    results = attack.execute_attack(data, llm)

    # Step 4: Evaluate
    rate = JailbreakRate(results).compute_metric()
    print(f"✅ Jailbreak Rate for {model_name}: {rate:.4f}")

    # Optional: Save results
    with open("results/jailbreak_ollama_results.csv", "a") as f:
        f.write(f"{model_name},Jailbreak,{rate:.4f}\n")
