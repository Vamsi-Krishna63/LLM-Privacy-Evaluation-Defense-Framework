import pandas as pd
import matplotlib.pyplot as plt

# Load each CSV with no header and assign proper column names
mia_df = pd.read_csv("results\mia_ollama_results.csv", header=None, names=["model", "attack_type", "score"])
dea_df = pd.read_csv("results\dea_ollama_results.csv", header=None, names=["model", "attack_type", "score"])
jailbreak_df = pd.read_csv("results\jailbreak_ollama_results.csv", header=None, names=["model", "attack_type", "score"])
prompt_leakage_df = pd.read_csv("results\prompt_leakage_results.csv", header=None, names=["model", "attack_type", "score"])

# Combine all into one DataFrame
all_attacks_df = pd.concat([mia_df, dea_df, jailbreak_df, prompt_leakage_df], ignore_index=True)

# Pivot to create model-wise comparison across attacks
attack_pivot_df = all_attacks_df.pivot(index="model", columns="attack_type", values="score").reset_index()

# Plot the comparison chart
df_plot = attack_pivot_df.set_index("model")

plt.figure(figsize=(12, 6))
bar_plot = df_plot.plot(
    kind="bar",
    figsize=(14, 6),
    colormap="Set2",  # Distinct colors for each attack
    edgecolor="black"
)

plt.title("LLM Vulnerability Comparison Across Attack Types", fontsize=16)
plt.ylabel("Attack Success Rate", fontsize=12)
plt.xlabel("LLM Model", fontsize=12)
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(title="Attack Type", bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10)
plt.tight_layout()
plt.show()
