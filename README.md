# Placement Compass

An end-to-end Flask application for student placement readiness assessment.

## Included

- Responsive placement dashboard and dataset explorer
- Profile-based placement probability with tailored improvement suggestions
- Salary-package estimate for placed students
- Model comparison using Logistic Regression, Decision Tree, Random Forest, KNN, SVM, Gradient Boosting, AdaBoost, Linear Regression, and Random Forest Regressor
- Safe preprocessing pipeline: missing-value imputation, categorical encoding, and numeric scaling

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in a browser.

## Note

The output is educational decision support based on the included historical dataset; it is not a placement guarantee.
