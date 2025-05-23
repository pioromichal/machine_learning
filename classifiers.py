import numpy as np
from collections import Counter
from typing import Tuple
from joblib import Parallel, delayed


class Classifier:
    def __init__(self):
        pass

    def predict(self, sample):
        pass

    def fit(self, data: np.ndarray, classes: np.ndarray):
        pass


class GaussianNaiveBayes(Classifier):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.priors = {}
        self.likelihoods = {}

    def fit(self, train_features, train_classes):
        classes = np.unique(train_classes)
        classes_count = Counter(train_classes)

        for clas in classes:
            self.priors[clas] = classes_count[clas] / len(train_classes)
            self.likelihoods[clas] = {}

            class_indices = np.where(train_classes == clas)[0]
            class_data = train_features[class_indices]

            for feature_idx in range(train_features.shape[1]):
                feature_values = class_data[:, feature_idx]
                mean = np.mean(feature_values)
                std = np.std(feature_values)
                self.likelihoods[clas][feature_idx] = {"mean": mean, "std": std}

    def normal_dist(self, x, mean, std):
        std = max(std, self.epsilon)
        return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-((x - mean) ** 2) / (2 * (std ** 2)))

    def predict(self, sample):
        predictions = {}
        for clas, prior in self.priors.items():
            value = prior
            for feature_idx, feature_value in enumerate(sample):
                mean = self.likelihoods[clas][feature_idx]["mean"]
                std = self.likelihoods[clas][feature_idx]["std"]
                value *= self.normal_dist(feature_value, mean, std)
            predictions[clas] = value
        return max(predictions, key=predictions.get)


class Group:
    def __init__(self, group_classes):
        self.group_classes = group_classes
        self.entropy = self.group_entropy()

    def __len__(self):
        return len(self.group_classes)

    def group_entropy(self):
        entropy = 0
        num_samples = len(self)
        # if num_samples == 0:
        #     return 0.0
        class_counts = Counter(self.group_classes)
        for class_count in class_counts.values():
            entropy += self._entropy_func(class_count, num_samples)
        return entropy

    @staticmethod
    def _entropy_func(class_count: int, num_samples: int) -> float:
        # if class_count == 0 or num_samples == 0:
        #     return 0.0
        probability = class_count / num_samples
        entropy = - probability * np.log(probability)
        return entropy


class Node:
    def __init__(self, split_feature=None, split_val=None, depth=None, child_node_a=None, child_node_b=None, val=None):
        self.split_feature = split_feature
        self.split_val = split_val
        self.depth = depth
        self.child_node_a = child_node_a
        self.child_node_b = child_node_b
        self.val = val

    def predict(self, data):
        if self.val is not None:
            return self.val
        elif data[self.split_feature] <= self.split_val:
            return self.child_node_a.predict(data)
        else:
            return self.child_node_b.predict(data)


class DecisionTreeClassifier(Classifier):
    def __init__(self, max_depth, max_features="sqrt"):
        super().__init__()
        self.depth = 0
        self.max_depth = max_depth
        self.max_features = max_features
        self.tree = None

    @staticmethod
    def get_split_entropy(group_a: Group, group_b: Group):
        split_entropy = 0
        total_count = len(group_a) + len(group_b)
        for group in [group_a, group_b]:
            split_entropy += (len(group) / total_count) * group.group_entropy()
        return split_entropy

    def get_information_gain(self, parent_group: Group, child_group_a: Group, child_group_b: Group) -> float:
        information_gain = parent_group.group_entropy() - self.get_split_entropy(child_group_a, child_group_b)
        return information_gain

    def get_best_feature_split(self, feature_values, classes):
        unique_feature_vals = np.unique(feature_values)
        best_split_val, best_gain = None, -np.inf
        parent_group = Group(classes)

        for feature_val in unique_feature_vals:
            child_a_ids = feature_values <= feature_val
            child_group_a = Group(classes[child_a_ids])
            child_group_b = Group(classes[~child_a_ids])
            if len(child_group_a) == 0 or len(child_group_b) == 0:
                continue
            gain = self.get_information_gain(parent_group, child_group_a, child_group_b)

            if gain > best_gain:
                best_split_val = feature_val
                best_gain = gain
        return best_split_val, best_gain

    def get_best_split(self, data: np.ndarray, classes: np.ndarray):
        best_feature, best_split_val, best_gain = None, None, -np.inf

        n_features = data.shape[1]

        if self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "all":
            k = n_features
        elif isinstance(self.max_features, int):
            k = min(self.max_features, n_features)
        else:
            raise ValueError("Invalid max_features value")

        features_to_try = np.random.choice(n_features, size=k, replace=False)

        for feature in features_to_try:
            split, gain = self.get_best_feature_split(data[:, feature], classes)

            if gain > best_gain:
                best_feature = feature
                best_split_val = split
                best_gain = gain

        if best_feature is None or best_split_val is None:
            return None, None

        return best_feature, best_split_val

    def _build_tree(self, data, classes, depth=0):
        if len(np.unique(classes)) == 1:
            return Node(val=classes[0])
        if depth == self.max_depth:
            return Node(val=np.bincount(classes).argmax())

        best_feature, best_split_val = self.get_best_split(data, classes)

        if best_feature is None or best_split_val is None:
            # Nie udało się znaleźć sensownego podziału — zwróć najczęstszą klasę
            return Node(val=np.bincount(classes).argmax())

        child_a_ids = data[:, best_feature] <= best_split_val
        child_classes_a = classes[child_a_ids]
        child_classes_b = classes[~child_a_ids]
        child_data_a = data[child_a_ids, :]
        child_data_b = data[~child_a_ids, :]

        child_node_a = self._build_tree(child_data_a, child_classes_a, depth + 1)
        child_node_b = self._build_tree(child_data_b, child_classes_b, depth + 1)

        return Node(
            split_feature=best_feature,
            split_val=best_split_val,
            depth=depth,
            child_node_a=child_node_a,
            child_node_b=child_node_b
        )

    def fit(self, data: np.ndarray, classes: np.ndarray):
        self.tree = self._build_tree(data, classes)

    def predict(self, data):
        return self.tree.predict(data)


class RandomForest(Classifier):
    def __init__(self, trees_number, tree_percentage=1.0, n_jobs=-1,
                 max_features="sqrt", epsilon=1e-6, discrete_x=True):
        super().__init__()
        self._trees_number = trees_number
        self._tree_percentage = tree_percentage
        self._n_jobs = n_jobs
        self._max_features = max_features
        self._epsilon = epsilon
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
        model = DecisionTreeClassifier(max_depth=1e10, max_features=self._max_features)
        x_i, y_i = self.bootstrap(x, y)
        model.fit(x_i, y_i)
        return model

    def _train_naive_bayes(self, x, y):
        model = GaussianNaiveBayes(epsilon=self._epsilon)
        x_i, y_i = self.bootstrap(x, y)
        model.fit(x_i, y_i)
        return model

    def _predict_sample(self, sample: np.ndarray) -> int:
        predictions = [model.predict(sample) for model in self._models]
        majority_vote = Counter(predictions).most_common(1)[0][0]
        return majority_vote

    def predict(self, data: np.ndarray) -> np.ndarray:
        return np.array([self._predict_sample(sample) for sample in data])
