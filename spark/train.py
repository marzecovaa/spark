import numpy as np
import joblib
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from spark.model_io import save_model

def train_base_model(
    X_train: np.ndarray,
    y_train:np.ndarray,
    random_state=42,
    max_iter=1000,
    n_splits=5,
    verbose=0) -> StackingClassifier:
    """ Fit the model and return a fitted model
    """
    #base models
    base_models = [
    ('svc', SVC(probability=True, class_weight='balanced', random_state=random_state)),
    ('catboost', CatBoostClassifier(verbose=verbose,auto_class_weights='Balanced',random_state=random_state)),
    ]
    #define meta model
    meta_model = LogisticRegression(
    class_weight='balanced',
    max_iter=max_iter,
    random_state=random_state
    )

    #stacking classifier
    stacked_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state),
    passthrough=True,
    n_jobs=-1
    )

    #Train
    trained_model = stacked_model.fit(X_train, y_train)

    # Save and return path
    model_path = save_model(trained_model)
    return model_path


def evaluate_base_model(
    model_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray) -> float:

    """
    Load & evaluate trained model from the given path on the test data set
    """
    # Load model from path
    model = joblib.load(model_path)

    y_pred = model.predict(X_test)
    base_score = round(balanced_accuracy_score(y_test, y_pred),2)

    print(f"✅ Model evaluated, Balanced Accuracy: {base_score}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    return base_score
