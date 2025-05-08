from skmultiflow.drift_detection import KSDDDetector
import pandas as pd

def generate_drift_report(reference_data, current_data, output_path="churn-prediction-app/data/drift_report.html"):
    # التحضير للكشف عن الانحراف باستخدام KSDDDetector
    ksdd = KSDDDetector()  # استخدام اختبار KS للكشف عن الانحراف

    drift_results = []
    
    # دمج البيانات المرجعية والبيانات الحالية
    combined_data = pd.concat([reference_data, current_data])

    # التحقق من الانحراف بين البيانات المرجعية والبيانات الحالية
    for column in combined_data.columns:
        ksdd.add_element(combined_data[column].values)  # إضافة البيانات الجديدة
        drift_detected = ksdd.detected_change  # التحقق من الكشف عن انحراف
        drift_results.append((column, drift_detected))

    # إنشاء تقرير الانحراف
    with open(output_path, 'w') as file:
        file.write("<html><body><h1>Data Drift Report</h1><ul>")
        for column, drift in drift_results:
            drift_status = "Detected" if drift else "Not Detected"
            file.write(f"<li>Column: {column} - Drift: {drift_status}</li>")
        file.write("</ul></body></html>")
    
    return output_path
