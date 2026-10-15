
import argparse
import os
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    GradientBoostingClassifier, AdaBoostClassifier,
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, AdaBoostRegressor
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
PLOTS = OUT / "plots"
REPORTS = OUT / "reports"
MODELS = OUT / "models"

for folder in (PLOTS, REPORTS, MODELS):
    folder.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")


def save_plot(name):
    plt.tight_layout()
    path = PLOTS / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved graph:", path)


def open_saved_graphs():
    graphs = sorted(PLOTS.glob("*.png"))

    print("\n========== OPENING GRAPHS ==========")
    if not graphs:
        print("No PNG graphs found.")
        return

    for i, graph in enumerate(graphs, start=1):
        try:
            print(f"Opening {i}/{len(graphs)}: {graph.name}")
            os.startfile(str(graph))
            time.sleep(0.4)
        except OSError as error:
            print(f"Could not open {graph.name}: {error}")

    print(f"\nRequested opening of {len(graphs)} graph files.")
    print("Graph folder:", PLOTS)


def load_data(path=None):
    if path:
        csv = Path(path)
        if not csv.is_file():
            raise FileNotFoundError(f"Dataset not found: {csv}")
    else:
        candidates = [
            ROOT / "placement_predict_50k Dataset.csv",
            ROOT / "placement_predict_50k_Dataset.csv",
            ROOT / "data" / "placement_predict_50k Dataset.csv",
        ]
        candidates = [p for p in candidates if p.is_file()]

        if not candidates:
            candidates = list(ROOT.glob("*placement*.csv"))
        if not candidates:
            candidates = list(ROOT.glob("*.csv"))

        if len(candidates) != 1:
            raise FileNotFoundError(
                "Dataset missing or multiple CSVs found. "
                'Run with --data "FULL_PATH_TO_CSV"'
            )
        csv = candidates[0]

    df = pd.read_csv(csv)
    df.columns = [str(c).strip() for c in df.columns]

    if df.empty:
        raise ValueError("Dataset is empty.")

    print("\nDataset:", csv)
    return df


def eda(df):
    print("\n========== DATA SUMMARY ==========")
    print("Shape:", df.shape)
    print("\nColumns:", df.columns.tolist())
    print("\nFirst rows:\n", df.head())
    print("\nData types:\n", df.dtypes)
    print("\nStatistics:\n", df.describe(include="all").T)
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())

    quality = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2),
        "unique": df.nunique(dropna=False)
    })
    quality.to_csv(REPORTS / "data_quality.csv")

    numeric = df.select_dtypes(include=np.number)

    if not numeric.empty:
        numeric.describe().T.to_csv(REPORTS / "numeric_statistics.csv")

        plt.figure(figsize=(12, 8))
        sns.heatmap(
            numeric.corr(), cmap="coolwarm", center=0,
            annot=len(numeric.columns) <= 12, fmt=".2f"
        )
        plt.title("Feature Correlation Heatmap")
        save_plot("01_correlation_heatmap.png")

        cols = list(numeric.columns[:15])

        rows = int(np.ceil(len(cols) / 3))
        fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
        axes = np.array(axes).reshape(-1)

        for ax, col in zip(axes, cols):
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution: {col}")

        for ax in axes[len(cols):]:
            ax.set_visible(False)

        save_plot("02_histograms.png")

        fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
        axes = np.array(axes).reshape(-1)

        for ax, col in zip(axes, cols):
            sns.boxplot(y=df[col], ax=ax)
            ax.set_title(f"Outliers: {col}")

        for ax in axes[len(cols):]:
            ax.set_visible(False)

        save_plot("03_boxplots.png")

        pairs = []
        for i in range(min(len(cols), 5)):
            for j in range(i + 1, min(len(cols), 5)):
                pairs.append((cols[i], cols[j]))

        if pairs:
            sample = df.sample(min(len(df), 3000), random_state=42)
            rows_scatter = int(np.ceil(len(pairs) / 2))
            fig, axes = plt.subplots(
                rows_scatter, 2, figsize=(13, 4 * rows_scatter)
            )
            axes = np.array(axes).reshape(-1)

            for ax, (x, y) in zip(axes, pairs):
                sns.scatterplot(data=sample, x=x, y=y, alpha=.5, ax=ax)
                ax.set_title(f"{x} vs {y}")

            for ax in axes[len(pairs):]:
                ax.set_visible(False)

            save_plot("04_scatter_plots.png")

    if "PlacementStatus" in df.columns:
        plt.figure(figsize=(7, 5))
        sns.countplot(data=df, x="PlacementStatus")
        plt.title("Placement Status Distribution")
        save_plot("05_placement_distribution.png")

    if "CGPA" in df.columns and "Salary Package" in df.columns:
        sample = df.sample(min(len(df), 5000), random_state=42)

        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=sample, x="CGPA", y="Salary Package",
            hue="PlacementStatus" if "PlacementStatus" in df.columns else None
        )
        plt.title("CGPA vs Salary Package")
        save_plot("06_cgpa_salary_scatter.png")

        if "PlacementStatus" in df.columns:
            plt.figure(figsize=(8, 5))
            sns.boxplot(data=df, x="PlacementStatus", y="Salary Package")
            plt.title("Salary by Placement Status")
            save_plot("07_salary_by_status.png")

    categorical = [
        c for c in df.select_dtypes(exclude=np.number).columns
        if df[c].nunique(dropna=True) <= 25
    ][:10]

    if categorical:
        rows = int(np.ceil(len(categorical) / 2))
        fig, axes = plt.subplots(rows, 2, figsize=(14, 4 * rows))
        axes = np.array(axes).reshape(-1)

        for ax, col in zip(axes, categorical):
            counts = df[col].astype(str).value_counts().head(15)
            sns.barplot(x=counts.values, y=counts.index, ax=ax)
            ax.set_title(f"Category Counts: {col}")

        for ax in axes[len(categorical):]:
            ax.set_visible(False)

        save_plot("08_categorical_counts.png")


def make_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(X, scale=False):
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []

    if numeric:
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps), numeric))

    if categorical:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", make_encoder())
            ]),
            categorical
        ))

    return ColumnTransformer(transformers=transformers)


def classification(df):
    target = "PlacementStatus"
    if target not in df.columns:
        print("PlacementStatus not found; classification skipped.")
        return

    data = df.loc[df[target].notna()].copy()
    data[target] = data[target].astype(str).str.strip()
    data = data[data[target] != ""]

    if data[target].nunique() < 2:
        print("Classification needs at least two target classes.")
        return

    excluded = {
        target, "Salary Package", "Salary", "Package",
        "PlacementSalary", "IsAnomaly"
    }

    features = [
        c for c in data.columns
        if c not in excluded
        and data[c].nunique(dropna=True) > 1
        and c.lower() not in {
            "id", "studentid", "student_id", "rollno", "roll_no"
        }
    ]

    if not features:
        print("No usable classification features.")
        return

    X, y = data[features], data[target]
    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=RANDOM_STATE, stratify=stratify
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
        "AdaBoost": AdaBoostClassifier(
            n_estimators=80, random_state=RANDOM_STATE
        ),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE)
    }

    results, fitted, predictions, probabilities = [], {}, {}, {}

    for name, model in models.items():
        try:
            pipe = Pipeline([
                ("prep", make_preprocessor(
                    X_train, scale=name in {
                        "Logistic Regression", "KNN", "SVM"
                    }
                )),
                ("model", model)
            ])

            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)

            fitted[name] = pipe
            predictions[name] = pred

            row = {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Balanced Accuracy": balanced_accuracy_score(y_test, pred),
                "Precision weighted": precision_score(
                    y_test, pred, average="weighted", zero_division=0
                ),
                "Recall weighted": recall_score(
                    y_test, pred, average="weighted", zero_division=0
                ),
                "F1 weighted": f1_score(
                    y_test, pred, average="weighted", zero_division=0
                )
            }

            if hasattr(pipe, "predict_proba"):
                prob = pipe.predict_proba(X_test)
                probabilities[name] = prob
                classes = list(pipe.named_steps["model"].classes_)

                if len(classes) == 2:
                    binary_y = (y_test == classes[1]).astype(int)
                    row["ROC AUC"] = roc_auc_score(binary_y, prob[:, 1])
                    row["Average Precision"] = average_precision_score(
                        binary_y, prob[:, 1]
                    )

            counts = y_train.value_counts()
            cv_splits = min(3, int(counts.min()))

            if cv_splits >= 2:
                try:
                    scores = cross_val_score(
                        pipe, X_train, y_train,
                        cv=cv_splits, scoring="f1_weighted", n_jobs=1
                    )
                    row["CV F1 mean"] = scores.mean()
                except Exception:
                    pass

            results.append(row)
            print(
                f"{name}: accuracy={row['Accuracy']:.4f}, "
                f"F1={row['F1 weighted']:.4f}"
            )

        except Exception as exc:
            print(f"Skipped {name}: {exc}")

    if not results:
        print("No classification model succeeded.")
        return

    result = pd.DataFrame(results).sort_values(
        ["F1 weighted", "Accuracy"], ascending=False
    )
    result.to_csv(REPORTS / "classification_model_comparison.csv", index=False)
    print("\nCLASSIFICATION COMPARISON\n", result.round(4).to_string(index=False))

    long = result.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Balanced Accuracy", "F1 weighted"],
        var_name="Metric", value_name="Score"
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(data=long, x="Model", y="Score", hue="Metric")
    plt.xticks(rotation=35)
    plt.ylim(0, 1)
    plt.title("Classification Model Comparison")
    save_plot("09_classification_model_comparison.png")

    best_name = result.iloc[0]["Model"]
    best = fitted[best_name]
    pred = predictions[best_name]

    report = classification_report(y_test, pred, zero_division=0)
    print("\nBEST CLASSIFIER:", best_name)
    print(report)

    (REPORTS / "classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    classes = list(best.named_steps["model"].classes_)
    cm = confusion_matrix(y_test, pred, labels=classes)
    ConfusionMatrixDisplay(
        cm, display_labels=classes
    ).plot(cmap="Blues", values_format="d")

    plt.title(f"Confusion Matrix: {best_name}")
    save_plot("10_confusion_matrix.png")

    if len(classes) == 2 and best_name in probabilities:
        prob = probabilities[best_name][:, 1]
        binary_y = (y_test == classes[1]).astype(int)

        fpr, tpr, _ = roc_curve(binary_y, prob)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(binary_y, prob):.3f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        save_plot("11_roc_curve.png")

        precision, recall, _ = precision_recall_curve(binary_y, prob)
        plt.figure(figsize=(7, 6))
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        save_plot("12_precision_recall_curve.png")

    try:
        prep = best.named_steps["prep"]
        model = best.named_steps["model"]
        names = prep.get_feature_names_out()

        if hasattr(model, "feature_importances_"):
            importance = pd.DataFrame({
                "Feature": names,
                "Importance": model.feature_importances_
            })
        else:
            perm = permutation_importance(
                best, X_test, y_test, n_repeats=2,
                random_state=RANDOM_STATE,
                scoring="f1_weighted", n_jobs=1
            )
            importance = pd.DataFrame({
                "Feature": X_test.columns,
                "Importance": perm.importances_mean
            })

        importance = importance.sort_values(
            "Importance", ascending=False
        ).head(25)

        importance.to_csv(
            REPORTS / "classification_feature_importance.csv", index=False
        )

        plt.figure(figsize=(10, 8))
        sns.barplot(data=importance, x="Importance", y="Feature")
        plt.title("Classification Feature Importance")
        save_plot("13_feature_importance.png")

    except Exception as exc:
        print("Feature importance skipped:", exc)

    pred_df = X_test.copy()
    pred_df["Actual"] = y_test
    pred_df["Predicted"] = pred
    pred_df.to_csv(REPORTS / "classification_predictions.csv", index=False)

    joblib.dump(best, MODELS / "best_placement_classifier.joblib")
    print("Saved classifier:", MODELS / "best_placement_classifier.joblib")


def regression(df):
    target = next(
        (c for c in [
            "Salary Package", "Salary", "Package", "PlacementSalary"
        ] if c in df.columns),
        None
    )

    if target is None:
        print("Salary target not found; regression skipped.")
        return

    data = df.copy()
    data[target] = pd.to_numeric(data[target], errors="coerce")
    data = data.dropna(subset=[target])

    if len(data) < 10 or data[target].nunique() < 2:
        print("Insufficient salary data.")
        return

    excluded = {
        target, "Salary Package", "Salary", "Package",
        "PlacementSalary", "PlacementStatus", "IsAnomaly"
    }

    features = [
        c for c in data.columns
        if c not in excluded
        and data[c].nunique(dropna=True) > 1
        and c.lower() not in {
            "id", "studentid", "student_id", "rollno", "roll_no"
        }
    ]

    if not features:
        print("No usable regression features.")
        return

    X, y = data[features], data[target].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.2, random_state=RANDOM_STATE
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=8, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            random_state=RANDOM_STATE
        ),
        "AdaBoost": AdaBoostRegressor(
            n_estimators=80, random_state=RANDOM_STATE
        )
    }

    rows, fitted, predictions = [], {}, {}

    for name, model in models.items():
        try:
            pipe = Pipeline([
                ("prep", make_preprocessor(
                    X_train,
                    scale=name in {"Linear Regression", "Ridge"}
                )),
                ("model", model)
            ])

            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            mse = mean_squared_error(y_test, pred)

            fitted[name] = pipe
            predictions[name] = pred

            rows.append({
                "Model": name,
                "MAE": mean_absolute_error(y_test, pred),
                "MSE": mse,
                "RMSE": np.sqrt(mse),
                "R2": r2_score(y_test, pred)
            })

            print(
                f"{name}: RMSE={np.sqrt(mse):.4f}, "
                f"R2={r2_score(y_test, pred):.4f}"
            )

        except Exception as exc:
            print(f"Skipped {name}: {exc}")

    if not rows:
        print("No regression model succeeded.")
        return

    result = pd.DataFrame(rows).sort_values("RMSE")
    result.to_csv(REPORTS / "regression_model_comparison.csv", index=False)
    print("\nREGRESSION COMPARISON\n", result.round(4).to_string(index=False))

    plt.figure(figsize=(11, 6))
    sns.barplot(data=result, x="Model", y="RMSE")
    plt.xticks(rotation=35)
    plt.title("Salary Model Comparison - Lower RMSE Is Better")
    save_plot("14_regression_model_comparison.png")

    best_name = result.iloc[0]["Model"]
    best = fitted[best_name]
    pred = predictions[best_name]

    print("\nBEST REGRESSOR:", best_name)
    print("MAE:", mean_absolute_error(y_test, pred))
    print("MSE:", mean_squared_error(y_test, pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
    print("R2:", r2_score(y_test, pred))

    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, pred, alpha=.5)
    low = min(y_test.min(), np.min(pred))
    high = max(y_test.max(), np.max(pred))
    plt.plot([low, high], [low, high], "--")
    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.title("Actual vs Predicted Salary")
    save_plot("15_actual_vs_predicted_salary.png")

    residual = y_test.to_numpy() - np.asarray(pred)

    plt.figure(figsize=(8, 5))
    plt.scatter(pred, residual, alpha=.5)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Salary")
    plt.ylabel("Residual")
    plt.title("Salary Residual Plot")
    save_plot("16_salary_residual_plot.png")

    plt.figure(figsize=(8, 5))
    sns.histplot(residual, kde=True)
    plt.title("Residual Distribution")
    save_plot("17_salary_residual_distribution.png")

    output = X_test.copy()
    output["Actual Salary"] = y_test
    output["Predicted Salary"] = pred
    output["Residual"] = residual
    output.to_csv(REPORTS / "salary_predictions.csv", index=False)

    joblib.dump(best, MODELS / "best_salary_regressor.joblib")
    print("Saved regressor:", MODELS / "best_salary_regressor.joblib")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None, help="Path to dataset CSV")
    parser.add_argument("--eda-only", action="store_true")
    args = parser.parse_args()

    df = load_data(args.data)
    eda(df)

    if not args.eda_only:
        classification(df)
        regression(df)

    print("\n========== ALL DONE ==========")
    print("Plots:", PLOTS)
    print("Reports:", REPORTS)
    print("Models:", MODELS)

    open_saved_graphs()


if __name__ == "__main__":
    main()
