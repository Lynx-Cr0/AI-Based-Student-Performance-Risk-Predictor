# AI-Based Student Performance Risk Predictor

An educational multi-class classification project using a feed-forward Artificial Neural Network (ANN) with Keras. It predicts **Low**, **Medium**, or **High** student performance from structured academic and study-related features.

## Dataset
- 15,000 rows and 12 original columns.
- Target: `performance_category` (Low, Medium, High).
- `student_id` is an identifier and is excluded from model inputs.
- Model inputs: `age, attendance_pct, study_hours_per_week, previous_score, assignment_completion_pct, internal_assessment_score, participation_score, sleep_hours_per_day, extracurricular_hours_per_week, failed_subjects`.
- No missing values or duplicate rows were found in the supplied dataset.
- Class counts: Low 4,950; Medium 5,100; High 4,950.

## Method
1. Stratified 60% train / 20% validation / 20% test split (implemented as 80% train+validation then 20% test; validation is 20% of the 80% training partition).
2. Median imputation and StandardScaler are fitted on training data only.
3. Logistic Regression provides a traditional baseline.
4. ANN: Input(10) → Dense(32, ReLU) → Dropout(0.15) → Dense(16, ReLU) → Dense(3, Softmax).
5. Optimizer: Adam; loss: sparse categorical cross-entropy; early stopping monitors validation loss.
6. Test set is held out from model fitting and used for final metrics.

## Test Results
| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.6920 | 0.6958 | 0.6920 | 0.6933 |
| ANN | 0.6873 | 0.6856 | 0.6873 | 0.6861 |

The baseline is a useful reference. In this run the ANN did not outperform Logistic Regression; report this transparently rather than claiming an improvement. The medium class is more difficult to distinguish, so inspect its class-wise precision/recall in the report.

## Setup
Use Python 3.10–3.12 recommended. In a new virtual environment:

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run training
The included model and preprocessing objects are already trained and saved under `models/`. To retrain from the CSV:

```bash
python src/train.py
```

## Run the prediction UI
```bash
streamlit run app.py
```

## Run the API
```bash
uvicorn api:app --reload
```
Open `http://127.0.0.1:8000/docs` to test `POST /predict`.

Example JSON:
```json
{
  "age": 20,
  "attendance_pct": 82,
  "study_hours_per_week": 16,
  "previous_score": 71,
  "assignment_completion_pct": 80,
  "internal_assessment_score": 68,
  "participation_score": 7,
  "sleep_hours_per_day": 7,
  "extracurricular_hours_per_week": 4,
  "failed_subjects": 0
}
```

## Project structure
```text
student-performance-ann/
├── data/student_performance_dataset.csv
├── models/student_performance.keras
├── models/preprocessing.joblib
├── models/logistic_regression.joblib
├── notebooks/ann_training.ipynb
├── plots/
├── reports/
├── src/train.py
├── src/predict.py
├── app.py
├── api.py
├── requirements.txt
└── README.md
```

## Important interpretation note
Predictions are statistical estimates learned from the supplied dataset. They must not be used as the sole basis for consequential academic decisions. Human review and context are necessary.
