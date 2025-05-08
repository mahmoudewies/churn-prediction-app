def generate_drift_report(reference_data, current_data, output_path="data/drift_report.html"):
    try:
        from evidently.report import Report
        from evidently.metrics import DataDriftPreset

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference_data, current_data=current_data)
        report.save_html(output_path)
        return output_path
    except ImportError as e:
        print(f"ImportError: {e}")  # هذا سوف يساعدنا في معرفة سبب الخطأ
        st.error(f"Error while importing libraries: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")  # هذا سيساعدنا في معرفة إذا كان هناك خطأ آخر
        st.error(f"An error occurred: {e}")
