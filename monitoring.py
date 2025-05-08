from evidently.report import Report
from evidently.metrics import DataDriftPreset
import pandas as pd

def generate_drift_report(reference_data, current_data, output_path="data/drift_report.html"):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html(output_path)
    return output_path
