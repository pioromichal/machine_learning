import numpy as np
from collections import Counter


class Classifier:
    def __init__(self):
        pass

    def predict(self, sample):
        pass

    def fit(self, data: np.ndarray, classes: np.ndarray):
        pass


class GaussianNaiveBayes(Classifier):
    def __init__(self):
        super().__init__()
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

    @staticmethod
    def normal_dist(x, mean, std):
        if std == 0:
            std = 1e-6
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
    def __init__(self, max_depth):
        super().__init__()
        self.depth = 0
        self.max_depth = max_depth
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

        for feature in range(data.shape[1]):
            split, gain = self.get_best_feature_split(data[:, feature], classes)

            if gain > best_gain:
                best_feature = feature
                best_split_val = split
                best_gain = gain
        return best_feature, best_split_val

    def _build_tree(self, data, classes, depth=0):
        if len(np.unique(classes)) == 1:
            return Node(val=classes[0])
        if depth == self.max_depth:
            return Node(val=np.bincount(classes).argmax())

        best_feature, best_split_val = self.get_best_split(data, classes)
        child_a_ids = data[:, best_feature] <= best_split_val
        child_classes_a = classes[child_a_ids]
        child_classes_b = classes[~child_a_ids]
        child_data_a = data[child_a_ids, :]
        child_data_b = data[~child_a_ids, :]

        child_node_a = self._build_tree(child_data_a, child_classes_a, depth + 1)
        child_node_b = self._build_tree(child_data_b, child_classes_b, depth + 1)

        return Node(split_feature=best_feature, split_val=best_split_val, depth=depth, child_node_a=child_node_a,
                    child_node_b=child_node_b)

    def fit(self, data: np.ndarray, classes: np.ndarray):
        self.tree = self._build_tree(data, classes)

    def predict(self, data):
        return self.tree.predict(data)
