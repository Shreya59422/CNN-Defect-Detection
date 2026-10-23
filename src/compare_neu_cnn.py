
from pathlib import Path

import pandas as pd


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================
# 2. ORIGINAL CNN RESULTS
# ==========================================

original_metrics = {

    "Model": "Original CNN",

    "Accuracy": 0.9529,

    "Weighted Precision": 0.9531,

    "Weighted Recall": 0.9529,

    "Weighted F1-score": 0.9525,

    "Test Images": 276

}


# ==========================================
# 3. IMPROVED CNN RESULTS
# ==========================================

improved_metrics = {

    "Model": "Improved CNN",

    "Accuracy": 0.9638,

    "Weighted Precision": 0.9641,

    "Weighted Recall": 0.9638,

    "Weighted F1-score": 0.9637,

    "Test Images": 276

}


# ==========================================
# 4. CREATE COMPARISON TABLE
# ==========================================

comparison = pd.DataFrame([

    original_metrics,

    improved_metrics

])


# ==========================================
# 5. CALCULATE IMPROVEMENT
# ==========================================

accuracy_improvement = (

    improved_metrics["Accuracy"]

    - original_metrics["Accuracy"]

) * 100


f1_improvement = (

    improved_metrics["Weighted F1-score"]

    - original_metrics["Weighted F1-score"]

) * 100


print("\n")

print("=" * 65)

print("       NEU-CLS CNN MODEL COMPARISON")

print("=" * 65)


print(

    comparison.to_string(

        index=False,

        float_format=lambda value: f"{value:.4f}"

    )

)


# ==========================================
# 6. DISPLAY IMPROVEMENT
# ==========================================

print("\n")

print("=" * 65)

print("       MODEL PERFORMANCE IMPROVEMENT")

print("=" * 65)


print(

    f"Accuracy improvement: "

    f"{accuracy_improvement:.2f} percentage points"

)


print(

    f"Weighted F1 improvement: "

    f"{f1_improvement:.2f} percentage points"

)


# ==========================================
# 7. SAVE COMPARISON
# ==========================================

output_file = (

    RESULTS_DIR

    / "neu_cnn_model_comparison.csv"

)


comparison.to_csv(

    output_file,

    index=False

)


print("\nComparison saved to:")

print(output_file)


print("\nComparison completed successfully.")
