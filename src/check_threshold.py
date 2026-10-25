
import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_csv("results/anomaly_test_predictions.csv")

good = df.loc[df["actual_label"] == 0, "anomaly_score"]
defective = df.loc[df["actual_label"] == 1, "anomaly_score"]

print("Good score range:")
print("Min:", good.min())
print("Max:", good.max())
print("Median:", good.median())

print("\nDefective score range:")
print("Min:", defective.min())
print("Max:", defective.max())
print("Median:", defective.median())

print("\nScore direction check:")
print(
    "Defective mean:",
    round(defective.mean(), 6)
)
print(
    "Good mean:",
    round(good.mean(), 6)
)

print("\nThresholds are diagnostic only; these test labels")
print("must not be used to select the final operating threshold.")