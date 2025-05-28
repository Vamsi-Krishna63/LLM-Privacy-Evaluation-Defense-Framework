import pandas as pd
import matplotlib.pyplot as plt

# Load the results CSV
df = pd.read_csv("results/mia_ollama_results.csv", names=["model", "attack_type", "accuracy"])

# Drop duplicate rows in case you re-run attacks
df = df.drop_duplicates(subset=["model", "attack_type"], keep='last')

# Plot
plt.figure(figsize=(10, 6))
plt.bar(df['model'], df['accuracy'])
plt.ylim(0, 1)
plt.title("Membership Inference Attack Accuracy per Model")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.grid(True, axis='y')
plt.tight_layout()

# Save & show
plt.savefig("results/mia_accuracy_plot.png")
plt.show()
