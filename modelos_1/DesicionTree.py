import numpy as np
import pandas as pd

class Node:
    def __init__(self, x=None, y=None, feature=None, threshold=None, features_list=None):
        self.x = x
        self.y = y
        self.left = None
        self.right = None
        self.feature = feature
        self.threshold = threshold
        self.features_list = features_list

    def is_leaf(self):
        return self.left is None and self.right is None

    def is_terminal(self):
        return len(np.unique(self.y)) == 1

    def sample_size(self):
        return len(self.y)

    def calculate_gini(self):
        if len(self.y) == 0:
            return 0
        label_counts = np.unique(self.y, return_counts=True)[1]
        probabilities = label_counts / np.sum(label_counts)
        return 1 - np.sum(probabilities ** 2)

    def get_value(self):
        values, counts = np.unique(self.y, return_counts=True)
        max_index = np.argmax(counts)
        return values[max_index]


class DescisionTree_:
    def __init__(self, x_train, y_train, part_random=False, number_of_features=None):
        self.x_train = x_train #Data de entrenamiento
        self.y_train = y_train #Labels de entrenamiento
        self.root = None #Raiz del arbol
        self.part_random = part_random #Escoger si cada nivel seleccionara los paramatros aleatoriamente
        self.minimun_sample_size = 10 # El minimo numero de muestras por nodo
        self.number_of_features = number_of_features #La candidad de features a escoger en el proceso aleatorio no tiene mayor singnicancia al trabajar con el modelo en general
        self.features_list = x_train.columns.tolist() if number_of_features else None

    def fit(self):
        self.root = self.insert_node(self.root, self.part_random)

    def get_New_feature_list(self, node):
        if node.feature is None:
            if not self.part_random:
                node.features_list = self.x_train.columns.values.tolist()
                return node.features_list

        if self.part_random:

            #print(self.number_of_features)
            return np.random.choice(self.features_list, int(self.number_of_features), replace=True)
        else:
            current_feature = node.feature
            filtered_features = [feature for feature in node.features_list if feature != current_feature]
            return filtered_features

    def insert_node(self, node, part_random):
        if node is None:
            # print("WE ARE ON ROOT \n")
            node = Node(self.x_train, self.y_train, features_list=self.x_train.columns.values.tolist())
            if self.part_random:
                self.features_list = self.x_train.columns.values.tolist()
        #print("CURRENT SAMPLE SIZE ", node.sample_size())
        if not node.is_terminal() and node.sample_size() >= self.minimun_sample_size and len(node.features_list) >= 1:
            node.features_list = self.get_New_feature_list(node)
            print("current feature list \n", len(node.features_list))
            best_feature, threshold = self.BestSplitNode(node)
            node.threshold = threshold
            node.feature = best_feature
            node = self.Assignleaf(node)
            # Split the features list for child nodes
            if node.left is not None and node.left.sample_size() > 0:
                node.left.features_list = self.get_New_feature_list(node)
                self.insert_node(node.left, part_random)
            if node.right is not None and node.right.sample_size() > 0:
                node.right.features_list = self.get_New_feature_list(node)
                self.insert_node(node.right, part_random)
        return node

    def Assignleaf(self, node):
        try:

            x_filtered_left = node.x[node.x[node.feature] <= node.threshold].reset_index(drop=True)
            x_filtered_right = node.x[node.x[node.feature] > node.threshold].reset_index(drop=True)
            y_filtered_left = node.y[node.x[node.feature] <= node.threshold]
            y_filtered_right = node.y[node.x[node.feature] > node.threshold]

            node.left = Node(x=x_filtered_left, y=y_filtered_left, feature=node.feature,
                             features_list=node.features_list)
            node.right = Node(x=x_filtered_right, y=y_filtered_right, feature=node.feature,
                              features_list=node.features_list)
        except Exception as e:
            print(f"Error in Assignleaf: {e}")
            print("Current feature:", node.feature)
            print("Current threshold:", node.threshold)

        return node

    def BestSplitNode(self, node):
        best_gini = float('inf')
        best_feature, best_threshold = None, None

        for feature in node.features_list:
            gini, threshold = self.Numerical_gini_index(node, feature)
            if gini < best_gini:
                best_gini, best_feature, best_threshold = gini, feature, threshold

        return best_feature, best_threshold

    def Numerical_gini_index(self, node, feature):
        feature_values = node.x[feature].to_numpy()
        labels = node.y

        if len(labels) == 0 or np.unique(feature_values).size < 2:
            return float('inf'), None

        indices = np.argsort(feature_values)
        feature_values, labels = feature_values[indices], labels[indices]

        best_gini = float('inf')
        best_threshold = None
        left_counts = np.zeros(len(np.unique(labels)), dtype=int)
        right_counts = np.bincount(np.searchsorted(np.unique(labels), labels), minlength=len(np.unique(labels)))

        for i in range(1, len(feature_values)):
            idx = np.searchsorted(np.unique(labels), labels[i - 1])
            left_counts[idx] += 1
            right_counts[idx] -= 1

            if feature_values[i] > feature_values[i - 1] and np.sum(left_counts) > 0 and np.sum(right_counts) > 0:
                left_gini = 1 - np.sum((left_counts / np.sum(left_counts)) ** 2)
                right_gini = 1 - np.sum((right_counts / np.sum(right_counts)) ** 2)
                gini = (left_gini * np.sum(left_counts) + right_gini * np.sum(right_counts)) / (
                        np.sum(left_counts) + np.sum(right_counts))

                if gini < best_gini:
                    best_gini, best_threshold = gini, (feature_values[i - 1] + feature_values[i]) / 2

        return best_gini, best_threshold

    def RecursivePrediction(self, x_value, node):

        #print("PREDICTING IN ", node.feature)
        if node.is_leaf():
            return node.get_value()

        current_feature = node.feature

        if x_value[current_feature].values <= node.threshold:
            return self.RecursivePrediction(x_value, node.left)
        elif x_value[current_feature].values > node.threshold:
            return self.RecursivePrediction(x_value, node.right)

    def predict(self, x_test):
        y_pred = []

        for index, row in x_test.iterrows():
            row_df = pd.DataFrame(row).transpose()
            y_pred.append(self.RecursivePrediction(row_df, self.root))

        return y_pred

    def print_tree(self, node, depth=0):
        indent = "    " * depth
        if node is not None:
            if node.is_leaf():
                print(
                    f"{indent}Leaf: Gini={node.calculate_gini():.2f}, Samples={node.sample_size()}, Value={node.get_value()}")
            else:
                print(
                    f"{indent}Node: Feature={node.feature} <= {node.threshold:.2f}, Gini={node.calculate_gini():.2f}, Samples={node.sample_size()}, Value={node.get_value()}")
                self.print_tree(node.left, depth + 1)
                self.print_tree(node.right, depth + 1)
