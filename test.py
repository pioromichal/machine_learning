from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree

from classifiers import GaussianNaiveBayes, DecisionTreeClassifier

def evaluate_all(X, y, dataset_name):
    print(f"\n=== Dataset: {dataset_name} ===")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # --- Twoje: Gaussian Naive Bayes ---
    my_gnb = GaussianNaiveBayes()
    my_gnb.fit(X_train, y_train)
    my_preds = [my_gnb.predict(x) for x in X_test]
    my_acc = accuracy_score(y_test, my_preds)
    print(f"Your Gaussian Naive Bayes accuracy: {my_acc:.2f}")

    # --- Biblioteka: GaussianNB ---
    skl_gnb = GaussianNB()
    skl_gnb.fit(X_train, y_train)
    skl_preds = skl_gnb.predict(X_test)
    skl_acc = accuracy_score(y_test, skl_preds)
    print(f"sklearn GaussianNB accuracy: {skl_acc:.2f}")

    # --- Twoje: Decision Tree ---
    my_dt = DecisionTreeClassifier(max_depth=3)
    my_dt.fit(X_train, y_train)
    my_dt_preds = [my_dt.predict(x) for x in X_test]
    my_dt_acc = accuracy_score(y_test, my_dt_preds)
    print(f"Your Decision Tree accuracy: {my_dt_acc:.2f}")

    # --- Biblioteka: DecisionTreeClassifier ---
    skl_dt = SklearnDecisionTree(criterion='entropy', max_depth=3)
    skl_dt.fit(X_train, y_train)
    skl_dt_preds = skl_dt.predict(X_test)
    skl_dt_acc = accuracy_score(y_test, skl_dt_preds)
    print(f"sklearn DecisionTree accuracy: {skl_dt_acc:.2f}")

# === Test na zbiorze Iris ===
iris = load_iris()
evaluate_all(iris.data, iris.target, "Iris")

# === Test na zbiorze Wine ===
wine = load_wine()
evaluate_all(wine.data, wine.target, "Wine")
