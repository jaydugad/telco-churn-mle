import json

import joblib
import matplotlib
matplotlib.use("Agg")  # save charts without opening a window
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from xgboost import XGBClassifier

from src.config import METRICS_PATH, MODEL_PATH, RANDOM_STATE, ROOT_DIR, TEST_SIZE
from src.data import load_data
from src.pipeline import build_pipeline

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
CV_SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1"}


def evaluate(pipeline, X_test, y_test, threshold=0.5) -> dict:
    """Score a fitted pipeline on the held-out test set."""
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "pr_auc": round(average_precision_score(y_test, proba), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "accuracy": round(accuracy_score(y_test, pred), 4),
    }


def main():
    X, y = load_data()

    # Split first: the test set is never touched during training or tuning
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # Ratio of negatives to positives, used to weight the minority class in XGBoost
    neg_pos_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            scale_pos_weight=neg_pos_ratio, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=1,
        ),
    }

    results, fitted = {}, {}

    # 1. Compare models with 5-fold cross-validation on the training set
    for name, model in candidates.items():
        pipe = build_pipeline(model)
        cv = cross_validate(pipe, X_train, y_train, cv=CV, scoring=CV_SCORING, n_jobs=-1)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        results[name] = {
            "cv_pr_auc": round(cv["test_pr_auc"].mean(), 4),
            "cv_roc_auc": round(cv["test_roc_auc"].mean(), 4),
            "cv_f1": round(cv["test_f1"].mean(), 4),
            "test": evaluate(pipe, X_test, y_test),
        }
        print(f"{name:22s} CV PR-AUC: {results[name]['cv_pr_auc']}")

    # 2. Tune XGBoost (the primary model) with randomized search on the full pipeline
    param_dist = {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 1.0],
        "model__min_child_weight": [1, 3, 5],
    }
    search = RandomizedSearchCV(
        build_pipeline(XGBClassifier(
            scale_pos_weight=neg_pos_ratio, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=1,
        )),
        param_distributions=param_dist,
        n_iter=25,
        scoring="average_precision",
        cv=CV,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    fitted["xgboost_tuned"] = search.best_estimator_
    results["xgboost_tuned"] = {
        "cv_pr_auc": round(search.best_score_, 4),
        "best_params": search.best_params_,
        "test": evaluate(search.best_estimator_, X_test, y_test),
    }
    print(f"{'xgboost_tuned':22s} CV PR-AUC: {results['xgboost_tuned']['cv_pr_auc']}")

    # 3. Pick the best model by CV PR-AUC (not test score, to avoid selection bias)
    best_name = max(results, key=lambda k: results[k]["cv_pr_auc"])
    best_pipeline = fitted[best_name]
    print(f"\nBest model: {best_name}")
    print("Test metrics:", results[best_name]["test"])

    # 4. Save the model, metrics and precision-recall curve
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)

    with open(METRICS_PATH, "w") as f:
        json.dump({"best_model": best_name, "results": results}, f, indent=2)

    PrecisionRecallDisplay.from_estimator(best_pipeline, X_test, y_test, name=best_name)
    plt.title("Precision-Recall curve (test set)")
    plt.savefig(ROOT_DIR / "reports" / "pr_curve.png", dpi=150, bbox_inches="tight")

    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()