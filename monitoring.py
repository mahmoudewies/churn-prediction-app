def generate_drift_report(reference_data, current_data, output_path="data/drift_report.html"):
    from evidently.report import Report
    from evidently.metrics import DataDriftPreset
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html(output_path)
    return output_path
