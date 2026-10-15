"""Placement Compass: end-to-end placement prediction dashboard."""
from functools import lru_cache
from pathlib import Path
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "placement_predict_50k Dataset.csv"
TARGET, SALARY = "PlacementStatus", "Salary Package"
DROP = ["StudentID", "IsAnomaly", TARGET, SALARY]
app = Flask(__name__)

@app.template_filter("format_number")
def format_number(value): return f"{value:,}"

@lru_cache(maxsize=1)
def dataset(): return pd.read_csv(DATA_FILE)

def binary(values): return values.astype(str).str.lower().isin(["1", "yes", "placed", "true"]).astype(int)
def features(df): return [c for c in df.columns if c not in DROP]
def prepare(df):
    cols = features(df); numeric = df[cols].select_dtypes(include="number").columns.tolist(); categorical = [c for c in cols if c not in numeric]
    return ColumnTransformer([("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric), ("category", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical)])

@lru_cache(maxsize=1)
def classifiers():
    df = dataset().sample(n=min(12000, len(dataset())), random_state=42); X, y = df[features(df)], binary(df[TARGET]); tx, vx, ty, vy = train_test_split(X, y, test_size=.2, random_state=42, stratify=y)
    algorithms = {"Logistic Regression": LogisticRegression(max_iter=1200, class_weight="balanced"), "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=42), "Random Forest": RandomForestClassifier(n_estimators=160, max_depth=12, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1), "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=21, weights="distance"), "Support Vector Machine": SVC(C=1, probability=True, class_weight="balanced", random_state=42), "Gradient Boosting": GradientBoostingClassifier(n_estimators=130, learning_rate=.07, max_depth=3, random_state=42), "AdaBoost": AdaBoostClassifier(n_estimators=120, learning_rate=.6, random_state=42)}
    models, scores = {}, []
    for name, algorithm in algorithms.items():
        model = Pipeline([("prepare", prepare(df)), ("model", algorithm)]); model.fit(tx, ty); pred = model.predict(vx); prob = model.predict_proba(vx)[:, 1]
        models[name] = model; scores.append({"name": name, "accuracy": round(accuracy_score(vy, pred)*100, 2), "f1": round(f1_score(vy, pred)*100, 2), "auc": round(roc_auc_score(vy, prob)*100, 2)})
    scores.sort(key=lambda x: (x["auc"], x["f1"]), reverse=True); return models, scores

@lru_cache(maxsize=1)
def salary_models():
    df = dataset(); usable = df[df[SALARY].notna() & (df[SALARY] > 0)].sample(n=min(12000, int((df[SALARY] > 0).sum())), random_state=42); tx, vx, ty, vy = train_test_split(usable[features(usable)], usable[SALARY], test_size=.2, random_state=42)
    models, scores = {}, []
    for name, algorithm in {"Linear Regression": LinearRegression(), "Random Forest Regressor": RandomForestRegressor(n_estimators=150, max_depth=14, random_state=42, n_jobs=-1)}.items():
        model = Pipeline([("prepare", prepare(usable)), ("model", algorithm)]); model.fit(tx, ty); pred = model.predict(vx); models[name] = model; scores.append({"name": name, "mae": round(mean_absolute_error(vy, pred), 2), "r2": round(r2_score(vy, pred)*100, 2)})
    scores.sort(key=lambda x: x["r2"], reverse=True); return models, scores

def fields(df):
    result = []
    for col in features(df):
        if pd.api.types.is_numeric_dtype(df[col]):
            values = df[col].dropna(); result.append({"name": col, "kind": "number", "min": round(float(values.quantile(.01)), 2), "max": round(float(values.quantile(.99)), 2), "value": round(float(values.median()), 2)})
        else:
            options = sorted(df[col].dropna().astype(str).unique()); result.append({"name": col, "kind": "select", "options": options, "value": options[0]})
    return result

@app.route("/")
def home():
    df = dataset(); return render_template("index.html", rows=len(df), columns=len(df.columns), placement_rate=round(binary(df[TARGET]).mean()*100, 1), salary_rows=int((df[SALARY]>0).sum()))
@app.route("/predict")
def predict_page(): return render_template("predict.html", fields=fields(dataset()))
@app.post("/api/predict")
def predict():
    df = dataset(); profile = {f["name"]: (float(request.form.get(f["name"], f["value"])) if f["kind"] == "number" else request.form.get(f["name"], f["value"])) for f in fields(df)}; student = pd.DataFrame([profile])
    models, ranks = classifiers(); name = ranks[0]["name"]; chance = float(models[name].predict_proba(student)[0, 1]); salary, salary_ranks = salary_models(); package = max(0, float(salary[salary_ranks[0]["name"]].predict(student)[0]))
    advice = []
    for key, cutoff, text in [("CGPA",7.5,"Strengthen your CGPA through consistent semester performance."),("CodingTestScore",70,"Practise coding problems and timed assessments each week."),("AptitudeTestScore",70,"Build aptitude speed with mock tests and error reviews."),("MockInterviewScore",70,"Schedule mock interviews to improve structure and confidence.")]:
        if float(profile.get(key, cutoff)) < cutoff: advice.append(text)
    if not advice: advice.append("Your profile is balanced. Focus on tailoring projects and applications to each role.")
    if chance < .5: advice.append("Prioritise internships, portfolio projects, and interview preparation before placement season.")
    return jsonify(placed=chance>=.5, probability=round(chance*100,1), confidence="High" if abs(chance-.5)>=.25 else "Moderate", salary=round(package,2), model=name, advice=advice[:3])
@app.route("/models")
def models_page():
    _, cs = classifiers(); _, rs = salary_models(); return render_template("models.html", classifier_scores=cs, regression_scores=rs, features=features(dataset()))
@app.route("/data")
def data_page():
    df=dataset(); missing=df.isna().sum(); return render_template("data.html", shape=df.shape, columns=df.dtypes.astype(str).to_dict(), preview=df.head(12).to_dict("records"), missing=missing[missing>0].to_dict())
if __name__ == "__main__": app.run(debug=True, port=5000)
