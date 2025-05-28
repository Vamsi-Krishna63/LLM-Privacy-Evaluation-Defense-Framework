import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV file
csv_path = r"C:\Users\vamsi\Downloads\LLM-PBE-main\LLM-PBE-main\results\jailbreak_ollama_results.csv"
df = pd.read_csv(csv_path, names=["Model", "Attack", "JailbreakRate"])

# Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(df["Model"], df["JailbreakRate"], color="skyblue")
plt.title("Jailbreak Rate Across Ollama Models", fontsize=16)
plt.xlabel("LLM Model", fontsize=12)
plt.ylabel("Jailbreak Rate", fontsize=12)
plt.ylim(0, 1)

# Add value labels on top
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.2f}", ha='center', va='bottom')

plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.show()
