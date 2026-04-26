import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier

# Trains and evaluates multiple ML classifiers (Random Forrest, Gradient Boosting, Hist Gradient Boosting, XGBoost)
# Outputs classification metrics and confusion matrices.

# -----------------------------
# Random Forest with GridSearch
# -----------------------------
def randomForrest():
    train = pd.read_csv('../../data/data_numeric_only.csv')

    x_train, x_test, y_train, y_test = train_test_split(
        train.drop(["Label"], axis=1),
        train["Label"],
        test_size=0.2,
        random_state=42
    )

    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [50 + i * 10 for i in range(10)],
        "oob_score": [True, False],
        "max_features": ["sqrt", None, 6],
        "min_impurity_decrease": [0.0, 0.001],
        "class_weight": ["balanced"]
    }

    clf = GridSearchCV(
        RandomForestClassifier(n_estimators=100, random_state=42),
        param_grid,
        scoring="recall",
        n_jobs=-1,
        cv=10,
        verbose=0
    )

    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)

    precision, recall, fscore, support = score(
        y_test, y_pred, pos_label=1, average='binary'
    )

    print(classification_report(y_test, y_pred))

    print(
        'Metric for Random Forest: Precision: {} | Recall: {} | Accuracy: {}'.format(
            round(precision, 3),
            round(recall, 3),
            round((y_pred == y_test).sum() / len(y_pred), 3)
        )
    )

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    precision = precision_score(y_test, y_pred, average='weighted')
    print("Precision:", precision)

    recall = recall_score(y_test, y_pred, average='weighted')
    print("Recall:", recall)

    f1 = f1_score(y_test, y_pred, average='weighted')
    print("F1-Score:", f1)

# ----------------------
# Gradient Boosting Tree
# ----------------------
def gradientBoostedTree():

    train = pd.read_csv('../../data/data_numeric_only.csv', dtype='unicode')

    t = pd.read_csv('../../data/data_numeric_only.csv', dtype='unicode')
    t = t.drop(['Label'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        t, train['Label'], test_size=0.5
    )

    gbt = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.2,
        max_depth=20,
        max_features=2
    )

    gbt_model = gbt.fit(X_train, y_train)

    y_pred = gbt_model.predict(X_test)

    precision, recall, fscore, support = score(
        y_test, y_pred, pos_label='1', average='binary'
    )

    print(classification_report(y_test, y_pred))

    print(
        'Metric for Gradient Boosted Tree: Precision: {} | Recall: {} | Accuracy: {}'.format(
            round(precision, 3),
            round(recall, 3),
            round((y_pred == y_test).sum() / len(y_pred), 3)
        )
    )

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    precision = precision_score(y_test, y_pred, average='weighted')
    print("Precision:", precision)

    recall = recall_score(y_test, y_pred, average='weighted')
    print("Recall:", recall)

    f1 = f1_score(y_test, y_pred, average='weighted')
    print("F1-Score:", f1)


# ----------------------
# Hist Gradient Boosting
# ----------------------
def histGradientBoostedTree():

    train = pd.read_csv('../../data/data_numeric_only.csv', dtype='unicode')

    t = train.drop(['Label'], axis=1)
    labels = train['Label'].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        t, labels, test_size=0.5, random_state=42
    )

    hgbt = HistGradientBoostingClassifier(
        max_depth=20,
        learning_rate=0.2,
        max_iter=500
    )

    hgbt_model = hgbt.fit(X_train, y_train)

    y_pred = hgbt_model.predict(X_test)

    precision, recall, fscore, support = score(
        y_test, y_pred, pos_label=1, average='binary'
    )

    print(classification_report(y_test, y_pred))

    print(
        'Metric for Hist Gradient Boosted Tree: Precision: {} | Recall: {} | Accuracy: {}'.format(
            round(precision, 3),
            round(recall, 3),
            round((y_pred == y_test).sum() / len(y_pred), 3)
        )
    )

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    precision = precision_score(y_test, y_pred, average='weighted')
    print("Precision:", precision)

    recall = recall_score(y_test, y_pred, average='weighted')
    print("Recall:", recall)

    f1 = f1_score(y_test, y_pred, average='weighted')
    print("F1-Score:", f1)

# -------
# XGBoost
# -------
def XGBoost():

    train = pd.read_csv('../../data/data_numeric_only.csv')

    t = train.drop(['Label'], axis=1)
    t = t.apply(pd.to_numeric, errors='coerce').fillna(0)

    labels = train['Label'].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        t, labels, test_size=0.5, random_state=42
    )

    xgb = XGBClassifier(
        n_estimators=100,
        learning_rate=0.2,
        max_depth=20,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss"
    )

    xgb_model = xgb.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)

    precision, recall, fscore, support = score(
        y_test, y_pred, pos_label=1, average='binary'
    )

    print(classification_report(y_test, y_pred))

    print(
        'Metric for XGBoost: Precision: {} | Recall: {} | Accuracy: {}'.format(
            round(precision, 3),
            round(recall, 3),
            round((y_pred == y_test).sum() / len(y_pred), 3)
        )
    )

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)

    precision = precision_score(y_test, y_pred, average='weighted')
    print("Precision:", precision)

    recall = recall_score(y_test, y_pred, average='weighted')
    print("Recall:", recall)

    f1 = f1_score(y_test, y_pred, average='weighted')
    print("F1-Score:", f1)

def main():
    print("\n===== RANDOM FOREST =====")
    randomForrest()

    print("\n===== GRADIENT BOOSTED TREE =====")
    gradientBoostedTree()

    print("\n===== HIST GRADIENT BOOSTED TREE =====")
    histGradientBoostedTree()

    print("\n===== XGBOOST =====")
    XGBoost()

if __name__ == "__main__":
    main()
