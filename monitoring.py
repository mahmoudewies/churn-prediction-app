def generate_drift_report(reference_data, current_data, output_path="churn-prediction-app/data/drift_report.html"):
    from alibi.detect import Drachen
    import json

    # تحويل البيانات إلى تنسيق مناسب
    reference_data = reference_data.to_numpy()
    current_data = current_data.to_numpy()

    # إنشاء كائن دراكن للكشف عن التشتت
    drift_detector = Drachen()
    
    # اكتشاف التشتت
    drift_results = drift_detector.fit(reference_data, current_data)

    # حفظ التقرير في ملف HTML
    with open(output_path, "w") as f:
        json.dump(drift_results, f)

    return output_path
