import streamlit as st
from src.predict import predict_student, FEATURES
st.set_page_config(page_title="Student Performance Predictor",page_icon="🎓")
st.title("AI-Based Student Performance Risk Predictor")
st.write("Enter the student's academic and study-related information.")
defaults={"age":20,"attendance_pct":78.0,"study_hours_per_week":15.0,"previous_score":67.0,
"assignment_completion_pct":75.0,"internal_assessment_score":65.0,"participation_score":6.0,
"sleep_hours_per_day":7.0,"extracurricular_hours_per_week":5.0,"failed_subjects":1}
labels={"age":"Age","attendance_pct":"Attendance (%)","study_hours_per_week":"Study hours/week",
"previous_score":"Previous score","assignment_completion_pct":"Assignment completion (%)",
"internal_assessment_score":"Internal assessment score","participation_score":"Participation score",
"sleep_hours_per_day":"Sleep hours/day","extracurricular_hours_per_week":"Extracurricular hours/week",
"failed_subjects":"Failed subjects"}
with st.form("student_form"):
    values={}
    cols=st.columns(2)
    for i,f in enumerate(FEATURES):
        with cols[i%2]:
            values[f]=st.number_input(labels.get(f,f),value=float(defaults.get(f,0)),step=1.0)
    submitted=st.form_submit_button("Predict performance")
if submitted:
    try:
        result=predict_student(values)
        st.subheader("Prediction: "+result["predicted_performance"])
        st.metric("Model confidence",f'{result["confidence"]*100:.1f}%')
        st.write("Class probabilities")
        st.bar_chart(result["probabilities"])
        st.caption("This is an educational prediction aid, not a definitive assessment of a student.")
    except Exception as e: st.error(str(e))
