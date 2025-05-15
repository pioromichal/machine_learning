import numpy as np
from typing import Tuple
from .classifiers import Classifier, GaussianNaiveBayes, DecisionTreeClassifier
from joblib import Parallel, delayed
from collections import Counter


class RandomForest(Classifier):
    def __init__(self, trees_number, tree_percentage=1.0, n_jobs=None, discrete_x=True, discretization_type=None):
        super().__init__()
        self._trees_number = trees_number
        self._tree_percentage = tree_percentage
        self._n_jobs = n_jobs
        self._discrete_x = discrete_x
        self._discretization_type = discretization_type
        self._models = []
        self._trained = False


    @staticmethod
    def bootstrap(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.random.randint(0, len(x), len(x))
        return x[indices], y[indices]

    def fit(self, data: np.ndarray, classes: np.ndarray, **kwargs) -> None:
        super().fit(data, classes)

        n_trees = int(self._tree_percentage * self._trees_number)
        n_bayes = self._trees_number - n_trees

        used_jobs = 1 if self._n_jobs is None else self._n_jobs

        tree_models = Parallel(n_jobs=used_jobs)(
            delayed(self._train_tree)(data, classes) for _ in range(n_trees)
        )
        bayes_models = Parallel(n_jobs=used_jobs)(
            delayed(self._train_naive_bayes)(data, classes) for _ in range(n_bayes)
        )

        self._models = tree_models + bayes_models
        self._trained = True

    def _train_tree(self, x, y):
        model = DecisionTreeClassifier(max_depth=1e10)
        x_i, y_i = self.bootstrap(x, y)
        model.fit(x_i, y_i)
        return model

    def _train_naive_bayes(self, x, y):
        model = GaussianNaiveBayes()
        x_i, y_i = self.bootstrap(x, y)
        model.fit(x_i, y_i)
        return model

    def _predict_sample(self, sample: np.ndarray) -> int:
        predictions = [model.predict(sample) for model in self._models]
        majority_vote = Counter(predictions).most_common(1)[0][0]
        return majority_vote

    def predict(self, data: np.ndarray) -> np.ndarray:
        return np.array([self._predict_sample(sample) for sample in data])
