from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree

from classifiers import GaussianNaiveBayes, DecisionTreeClassifier


def evaluate_all(X, y, dataset_name, n_tests):
    print(f"\n=== Dataset: {dataset_name} ===")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    my_nbc_acc_sum = 0
    for _ in range(n_tests):
        my_gnb = GaussianNaiveBayes()
        my_gnb.fit(X_train, y_train)
        my_preds = [my_gnb.predict(x) for x in X_test]
        my_nbc_acc_sum += accuracy_score(y_test, my_preds)
    print(f"Average my NBC accuracy: {my_nbc_acc_sum/n_tests:.2f}")

    skl_nbc_acc_sum = 0
    for _ in range(n_tests):
        skl_gnb = GaussianNB()
        skl_gnb.fit(X_train, y_train)
        skl_preds = skl_gnb.predict(X_test)
        skl_nbc_acc_sum += accuracy_score(y_test, skl_preds)
    print(f"Average sklearn GaussianNB accuracy: {skl_nbc_acc_sum/n_tests:.2f}")

    my_dt_acc_sum = 0
    for _ in range(n_tests):
        my_dt = DecisionTreeClassifier(max_depth=3)
        my_dt.fit(X_train, y_train)
        my_dt_preds = [my_dt.predict(x) for x in X_test]
        my_dt_acc_sum += accuracy_score(y_test, my_dt_preds)
    print(f"Average my ID3 accuracy: {my_dt_acc_sum/n_tests:.2f}")

    skl_dt_acc_sum = 0
    for _ in range(n_tests):
        skl_dt = SklearnDecisionTree(criterion='entropy', max_depth=3)
        skl_dt.fit(X_train, y_train)
        skl_dt_preds = skl_dt.predict(X_test)
        skl_dt_acc_sum += accuracy_score(y_test, skl_dt_preds)
    print(f"Average sklearn ID3 accuracy: {skl_dt_acc_sum/n_tests:.2f}")

iris = load_iris()
evaluate_all(iris.data, iris.target, "Iris", n_tests=40)

wine = load_wine()
evaluate_all(wine.data, wine.target, "Wine", n_tests=40)
