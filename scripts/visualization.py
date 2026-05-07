import matplotlib.pyplot as plt
import pandas as pd
import os

def parse_validations(file_path):
    steps = []
    ppls = []
    if not os.path.exists(file_path):
        print(f"cant find {file_path}")
        return None
    
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("Steps:"):
                parts = line.split('\t')
                step = int(parts[0].replace("Steps: ", "").strip())
                ppl_val = None
                for p in parts:
                    if "ppl:" in p:
                        ppl_val = float(p.replace("ppl: ", "").strip())
                if step is not None and ppl_val is not None:
                    steps.append(step)
                    ppls.append(ppl_val)
    return pd.Series(index=steps, data=ppls)

base_path = "models"
models = {
    "Baseline (Regular)": os.path.join(base_path, "deen_transformer_regular", "validations.txt"),
    "Pre-norm": os.path.join(base_path, "deen_transformer_pre", "validations.txt"),
    "Post-norm": os.path.join(base_path, "deen_transformer_post", "validations.txt")
}

df_list = []
for name, path in models.items():
    series = parse_validations(path)
    if series is not None:
        series.name = name
        df_list.append(series)

results_df = pd.concat(df_list, axis=1)
results_df.index.name = "Steps"

results_df.to_csv("validation_ppl_comparison.csv")

# line chart
plt.figure(figsize=(10, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 

for i, col in enumerate(results_df.columns):
    valid_data = results_df[col].dropna()
    plt.plot(valid_data.index, valid_data.values, label=col, 
             color=colors[i], marker='o', markersize=4, linewidth=2)

plt.title("Validation Perplexity Comparison (Pre-norm vs Post-norm)")
plt.xlabel("Training Steps")
plt.ylabel("Perplexity (PPL)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.savefig("ppl_visualization.png", dpi=300)
plt.show()
