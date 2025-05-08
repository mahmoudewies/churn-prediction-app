from river import metrics
from river import drift
import pandas as pd

# Load your reference data and current data (for drift detection)
reference_data = pd.read_csv("data/reference_data.csv")
current_data = pd.read_csv("data/current_data.csv")

# إنشاء نموذج لاكتشاف تغير البيانات
drift_detector = drift.ADWIN()  # ADWIN هو خوارزم للكشف عن drift

# إظهار تقرير الكشوفات
def generate_drift_report(reference_data, current_data, output_path="data/drift_report.txt"):
    with open(output_path, "w") as file:
        for col in reference_data.columns:
            reference_column = reference_data[col]
            current_column = current_data[col]

            drift_score = drift_detector.update(reference_column, current_column)
            file.write(f"Column: {col}, Drift Detected: {drift_score > 0.5}\n")

    return output_path
