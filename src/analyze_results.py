
import pandas as pd
from pathlib import Path

path = Path("results/anomaly_test_predictions.csv")
df = pd.read_csv(path)

print("\nAnomaly scores by category:\n")

summary = df.groupby("category")["anomaly_score"].agg(
    ["count", "mean", "median", "min", "max", "std"]
)

print(summary.round(6))

print("\nLowest-scoring defective images:")
print(
    df[df["actual_label"] == 1]
    .nsmallest(10, "anomaly_score")[
        ["image", "category", "anomaly_score"]
    ].to_string(index=False)
)

print("\nHighest-scoring good images:")
print(
    df[df["actual_label"] == 0]
    .nlargest(10, "anomaly_score")[
        ["image", "category", "anomaly_score"]
    ].to_string(index=False)
)
