import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# =========================================================
# NORMALIZATION
# =========================================================

def normalize(df):
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["PlacementStatus"]
    )

    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)

    train_df = pd.DataFrame(
        train_scaled,
        columns=train_df.columns,
        index=train_df.index
    )

    test_df = pd.DataFrame(
        test_scaled,
        columns=test_df.columns,
        index=test_df.index
    )

    return train_df, test_df


def normalization(df):
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )

    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(train_df)
    test_scaled = scaler.transform(test_df)

    train_df = pd.DataFrame(
        train_scaled,
        columns=train_df.columns,
        index=train_df.index
    )

    test_df = pd.DataFrame(
        test_scaled,
        columns=test_df.columns,
        index=test_df.index
    )

    return train_df, test_df


# =========================================================
# STANDARDIZATION
# =========================================================

def standardization(df):

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )

    num_cols = [
        "CGPA",
        "Internships",
        "Projects",
        "Workshops"
    ]

    # Only use columns that actually exist
    num_cols = [
        col for col in num_cols
        if col in df.columns
    ]

    if not num_cols:
        return pd.DataFrame(), pd.DataFrame()

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        train_df[num_cols]
    )

    X_test_scaled = scaler.transform(
        test_df[num_cols]
    )

    train_scaled_df = pd.DataFrame(
        X_train_scaled,
        columns=num_cols,
        index=train_df.index
    )

    test_scaled_df = pd.DataFrame(
        X_test_scaled,
        columns=num_cols,
        index=test_df.index
    )

    return train_scaled_df, test_scaled_df


# =========================================================
# ONE HOT ENCODING
# =========================================================

def onehot(df):

    nominal_cols = ["Gender"]

    nominal_cols = [
        col for col in nominal_cols
        if col in df.columns
    ]

    if not nominal_cols:
        return pd.DataFrame(), pd.DataFrame()

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["PlacementStatus"]
    )

    ohe = OneHotEncoder(
        drop="first",
        sparse_output=False,
        handle_unknown="ignore"
    )

    train_ohe = ohe.fit_transform(
        train_df[nominal_cols]
    )

    test_ohe = ohe.transform(
        test_df[nominal_cols]
    )

    feature_names = ohe.get_feature_names_out(
        nominal_cols
    )

    train_ohe_df = pd.DataFrame(
        train_ohe,
        columns=feature_names,
        index=train_df.index
    )

    test_ohe_df = pd.DataFrame(
        test_ohe,
        columns=feature_names,
        index=test_df.index
    )

    return train_ohe_df, test_ohe_df


# =========================================================
# ORDINAL ENCODING
# =========================================================

def ordinal(df):

    df = df.copy()

    if "CGPA" not in df.columns:
        return df.copy(), df.copy()

    df["CGPA_Category"] = pd.cut(
        df["CGPA"],
        bins=[0, 6, 8, 10],
        labels=["low", "medium", "high"],
        ordered=True
    )

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["PlacementStatus"]
    )

    encoder = OrdinalEncoder(
        categories=[["low", "medium", "high"]],
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    train_df_encoded = train_df.copy()
    test_df_encoded = test_df.copy()

    train_df_encoded["CGPA_Category"] = encoder.fit_transform(
        train_df[["CGPA_Category"]]
    )

    test_df_encoded["CGPA_Category"] = encoder.transform(
        test_df[["CGPA_Category"]]
    )

    return train_df_encoded, test_df_encoded


# =========================================================
# LABEL ENCODING
# =========================================================

def label_encode_branch(df):

    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["PlacementStatus"]
    )

    encoder = LabelEncoder()

    train_df_encoded = train_df.copy()
    test_df_encoded = test_df.copy()

    if "Stream" in df.columns:

        train_df_encoded["Stream"] = encoder.fit_transform(
            train_df["Stream"]
        )

        test_df_encoded["Stream"] = encoder.transform(
            test_df["Stream"]
        )

        mapping = dict(
            zip(
                encoder.classes_,
                encoder.transform(encoder.classes_)
            )
        )

    else:
        mapping = {}

    return train_df_encoded, test_df_encoded, mapping


# =========================================================
# CGPA CATEGORY
# =========================================================

def create_cgpa_category(df):

    df = df.copy()

    df["CGPA_Category"] = pd.cut(
        df["CGPA"],
        bins=[0, 6, 8, 10],
        labels=["low", "medium", "high"],
        ordered=True
    )

    return df


# =========================================================
# NULL VALUES
# =========================================================

def find_null_values(df, column):

    null_values = df[column].isnull()

    print(
        "Number of NULL values:",
        null_values.sum()
    )

    print("\nRows where value is NULL:")

    print(df[null_values])


# =========================================================
# DATA CLEANING
# =========================================================

def data_cleaning(df):

    df = df.copy()

    if "Workshops" in df.columns:
        df["Workshops"] = df["Workshops"].fillna(
            df["Workshops"].mean()
        )

    if "AptitudeTestScore" in df.columns:
        df["AptitudeTestScore"] = df[
            "AptitudeTestScore"
        ].fillna(
            df["AptitudeTestScore"].median()
        )

    numerical_cols = [
        "CGPA",
        "AttendancePercent",
        "Internships",
        "AptitudeTestScore",
        "CodingTestScore",
        "MockInterviewScore"
    ]

    numerical_cols = [
        col for col in numerical_cols
        if col in df.columns
    ]

    if numerical_cols:

        imputer = KNNImputer(
            n_neighbors=3
        )

        df[numerical_cols] = imputer.fit_transform(
            df[numerical_cols]
        )

    return df


# =========================================================
# NUMERICAL FEATURES
# =========================================================

def get_numerical_features(df):

    num_cols = df.select_dtypes(
        include="number"
    ).columns

    return df[num_cols]


# =========================================================
# CATEGORICAL FEATURES
# =========================================================

def get_categorical_features(df):

    cat_cols = df.select_dtypes(
        include="object"
    ).columns

    return df[cat_cols]


# =========================================================
# OUTLIERS
# =========================================================

def detect_outliers(df, column):

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    return outliers


# =========================================================
# MODEL FEATURES
# =========================================================

def select_model_features(df):

    features = [
        "CGPA",
        "Internships",
        "Projects",
        "CodingTestScore"
    ]

    features = [
        col for col in features
        if col in df.columns
    ]

    X = df[features]
    y = df["PlacementStatus"]

    return X, y


# =========================================================
# SIMPLE LINEAR REGRESSION
# =========================================================

def simple_linear_regression(df):

    if "CGPA" not in df.columns or "Salary Package" not in df.columns:
        return None

    data = df[
        ["CGPA", "Salary Package"]
    ].dropna()

    X = data[["CGPA"]]
    y = data["Salary Package"]

    train_X, test_X, train_y, test_y = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = LinearRegression()

    model.fit(
        train_X,
        train_y
    )

    y_pred = model.predict(test_X)

    mse = mean_squared_error(
        test_y,
        y_pred
    )

    r2 = r2_score(
        test_y,
        y_pred
    )

    return {
        "MSE": float(mse),
        "R2": float(r2)
    }


# =========================================================
# KNN IMPUTATION
# =========================================================

def knn_impute(df, n_neighbors=3):

    filled = df.copy()

    before = int(
        filled.isnull().sum().sum()
    )

    if "Workshops" in filled.columns:

        filled["Workshops"] = filled[
            "Workshops"
        ].fillna(
            filled["Workshops"].mean()
        )

    if "AptitudeTestScore" in filled.columns:

        filled["AptitudeTestScore"] = filled[
            "AptitudeTestScore"
        ].fillna(
            filled["AptitudeTestScore"].median()
        )

    numerical_cols = [
        "CGPA",
        "AttendancePercent",
        "Internships",
        "AptitudeTestScore",
        "CodingTestScore",
        "MockInterviewScore"
    ]

    numerical_cols = [
        col for col in numerical_cols
        if col in filled.columns
    ]

    if numerical_cols:

        imputer = KNNImputer(
            n_neighbors=n_neighbors
        )

        filled[numerical_cols] = imputer.fit_transform(
            filled[numerical_cols]
        )

    after = int(
        filled.isnull().sum().sum()
    )

    return {
        "before": before,
        "after": after,
        "n_neighbors": n_neighbors,
        "columns": numerical_cols,
        "df": filled
    }


# =========================================================
# GRADIENT DESCENT
# =========================================================

def gradient_descent(
    X,
    y,
    learning_rate=0.001,
    iterations=100
):

    X = np.asarray(
        X,
        dtype=float
    ).ravel()

    y = np.asarray(
        y,
        dtype=float
    ).ravel()

    m = len(y)

    theta0 = 0.0
    theta1 = 0.0

    mse_history = []

    for _ in range(iterations):

        y_pred = theta0 + theta1 * X

        error = y_pred - y

        mse = np.mean(
            error ** 2
        )

        mse_history.append(mse)

        d_theta0 = (
            2 / m
        ) * np.sum(error)

        d_theta1 = (
            2 / m
        ) * np.sum(
            error * X
        )

        theta0 -= (
            learning_rate * d_theta0
        )

        theta1 -= (
            learning_rate * d_theta1
        )

    return (
        theta0,
        theta1,
        mse_history
    )


# =========================================================
# LINEAR REGRESSION USING GRADIENT DESCENT
# =========================================================

def simple_linear_regression_GD(df):

    if "CGPA" not in df.columns:
        return None

    if "PlacementStatus" not in df.columns:
        return None

    data = df[
        ["CGPA", "PlacementStatus"]
    ].dropna()

    X = data[["CGPA"]]

    y = data["PlacementStatus"]

    # Convert Yes/No placement status to numbers
    if y.dtype == "object":

        y = (
            y.astype(str)
            .str.lower()
            .map({
                "yes": 1,
                "no": 0,
                "placed": 1,
                "not placed": 0
            })
        )

        valid = y.notna()

        X = X[valid]
        y = y[valid]

    train_X, test_X, train_y, test_y = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    theta0, theta1, mse_history = gradient_descent(
        train_X,
        train_y,
        learning_rate=0.001,
        iterations=100
    )

    y_pred = (
        theta0 +
        theta1 * test_X["CGPA"].values
    )

    mse = mean_squared_error(
        test_y,
        y_pred
    )

    r2 = r2_score(
        test_y,
        y_pred
    )

    return {
        "theta0": float(theta0),
        "theta1": float(theta1),
        "MSE": float(mse),
        "R2": float(r2),
        "mse_history": mse_history
    }


# =========================================================
# MAIN REGRESSION FUNCTION
# =========================================================

def run_regression_models(df):

    target = "Salary Package"

    if target not in df.columns:

        return {
            "error": "Salary Package column not found",
            "models": {}
        }

    features = [
        "CGPA",
        "AttendancePercent",
        "Internships",
        "Projects",
        "CodingTestScore",
        "MockInterviewScore"
    ]

    features = [
        col for col in features
        if col in df.columns
    ]

    if not features:

        return {
            "error": "No regression features found",
            "models": {}
        }

    data = df[
        features + [target]
    ].copy()

    # Convert target to numeric
    data[target] = pd.to_numeric(
        data[target],
        errors="coerce"
    )

    data = data.dropna()

    if len(data) < 10:

        return {
            "error": "Not enough data for regression",
            "models": {}
        }

    X = data[features]

    y = data[target]

    train_X, test_X, train_y, test_y = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    models = {

        "Linear Regression":
            LinearRegression(),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
    }

    results = {}

    for name, model in models.items():

        model.fit(
            train_X,
            train_y
        )

        y_pred = model.predict(
            test_X
        )

        mse = mean_squared_error(
            test_y,
            y_pred
        )

        rmse = np.sqrt(mse)

        mae = mean_absolute_error(
            test_y,
            y_pred
        )

        r2 = r2_score(
            test_y,
            y_pred
        )

        results[name] = {

            "MSE":
                round(float(mse), 4),

            "RMSE":
                round(float(rmse), 4),

            "MAE":
                round(float(mae), 4),

            "R2":
                round(float(r2), 4)
        }

    return {
        "features": features,
        "target": target,
        "models": results,
        "train_rows": len(train_X),
        "test_rows": len(test_X)
    }