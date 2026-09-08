import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data_dir = os.path.join(os.path.dirname(__file__), "../data")
df = pd.read_csv(os.path.join(data_dir, "probe_results.csv"))

plt.plot(df["layer"], df["auroc"], marker="o", label="AUROC")
plt.plot(df["layer"], df["accuracy"], marker="s", label="Accuracy")
plt.xlabel("Layer")
plt.legend()
plt.savefig(os.path.join(data_dir, "layer_auroc_plot.png"))