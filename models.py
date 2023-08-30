import numpy as np
from graphviz import Digraph

################################## kNN ##################################
class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)
    
    def _predict(self, x):
        # Compute Euclidean distances
        distances = [np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self.X_train]
        # Sort the k nearest neighbours
        k_indices = np.argsort(distances)[:self.k]
        # Extract their labels
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # NOTE: in case of tie chooses 0 -> -1 by default of max() function but since
        # -1 is also the most common label in the dataset it is ok.
        most_common = max(k_nearest_labels, key=k_nearest_labels.count)
        return most_common

################################## TREE ##################################
class Node():
    def __init__(self, feature=None, threshold=None, left=None, right=None, gain=None, value=None):
        self.feature = feature # The feature used for splitting at this node. Defaults to None.
        self.threshold = threshold # The threshold used for splitting at this node. Defaults to None.
        self.left = left # The left child node. Defaults to None.
        self.right = right # The right child node. Defaults to None.
        self.gain = gain # The gain of the split. Defaults to None.
        self.value = value # If this node is a leaf node, this attribute represents the predicted value for the target variable. Defaults to None.

class DecisionTree():
    def __init__(self, min_samples=2, max_depth=2):
        self.min_samples = min_samples # Minimum number of samples required to split an internal node.
        self.max_depth = max_depth

    def split_data(self, dataset, feature, threshold):
    # Splits the given dataset into two datasets based on the given feature and threshold.
    # feature: Index of the feature to be split on.

        # Create empty arrays to store the left and right datasets
        left_dataset = []
        right_dataset = []
        
        # Loop over each row in the dataset and split based on the given feature and threshold
        for row in dataset:
            if row[feature] <= threshold:
                left_dataset.append(row)
            else:
                right_dataset.append(row)

        # Convert the left and right datasets to numpy arrays and return
        left_dataset = np.array(left_dataset)
        right_dataset = np.array(right_dataset)
        return left_dataset, right_dataset

    def entropy(self, y):
        entropy = 0

        # Loop over the values of y = {-1, 1}
        labels = np.unique(y)
        for label in labels:
            # Find the examples in y that have the current label
            label_examples = y[y == label]
            # Calculate the frequency of the current label in y: P(y=label)
            p = len(label_examples) / len(y) # p = \frac{N_{l^+}}{N_l}
            # Calculate the Scaled entropy using the current label and ratio
            entropy += -p * np.log2(p) / 2

        # Return the final entropy value
        return entropy

    def information_gain(self, parent, left, right):
    # Gain in terms of reduction of the entropy after a split

        information_gain = 0
        # Compute entropy for parent
        parent_entropy = self.entropy(parent)
        # Calculate weight for left and right nodes
        weight_left = len(left) / len(parent)
        weight_right= len(right) / len(parent)
        # Compute entropy for left and right nodes
        entropy_left, entropy_right = self.entropy(left), self.entropy(right)
        # Calculate weighted entropy 
        weighted_entropy = weight_left * entropy_left + weight_right * entropy_right
        # calculate information gain 
        information_gain = parent_entropy - weighted_entropy
        return information_gain

    
    def best_split(self, dataset, num_samples, num_features):
    # Finds the best split for the given dataset.

        # Dictionary to store the best split values
        best_split = {'gain':- 1, 'feature': None, 'threshold': None}
        # Loop over all the features
        for feature_index in range(num_features):
            # Get the feature at the current feature_index
            feature_values = dataset[:, feature_index]
            # Get unique values of that feature
            thresholds = np.unique(feature_values)
            # Loop over all values of the feature
            for threshold in thresholds:
                # Get left and right datasets
                left_dataset, right_dataset = self.split_data(dataset, feature_index, threshold)
                # Check if either datasets is empty
                if len(left_dataset) and len(right_dataset):
                    # Get y values of the parent and left, right nodes
                    y, left_y, right_y = dataset[:, -1], left_dataset[:, -1], right_dataset[:, -1]
                    # Compute information gain based on the y values
                    information_gain = self.information_gain(y, left_y, right_y)
                    # Update the best split if conditions are met
                    if information_gain > best_split["gain"]:
                        best_split["feature"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["left_dataset"] = left_dataset
                        best_split["right_dataset"] = right_dataset
                        best_split["gain"] = information_gain
        return best_split #  A dictionary with the best split feature index, threshold, gain, left and right datasets.

    
    def calculate_leaf_value(self, y):

        y = list(y)
        # Get the highest present class in the array
        most_occuring_value = max(y, key=y.count)
        return most_occuring_value
    
    def build_tree(self, dataset, current_depth=0):
    # Recursively builds a decision tree from the given dataset.
      
        # Split the dataset into X, y values
        X, y = dataset[:, :-1], dataset[:, -1]
        n_samples, n_features = X.shape
        # Keeps spliting until stopping conditions are met
        if n_samples >= self.min_samples and current_depth <= self.max_depth:
            # Get the best split
            best_split = self.best_split(dataset, n_samples, n_features)
            # Check if gain isn't zero
            if best_split["gain"] > 0: # A split cannot increase the entropy
                # Continue splitting the left and the right child. Increment current depth
                left_node = self.build_tree(best_split["left_dataset"], current_depth + 1)
                right_node = self.build_tree(best_split["right_dataset"], current_depth + 1)
                # Return decision node
                return Node(best_split["feature"], best_split["threshold"],
                            left_node, right_node, best_split["gain"])

        # Compute leaf node value
        leaf_value = self.calculate_leaf_value(y)
        # Return leaf node value
        return Node(value=leaf_value)
    
    def fit(self, X, y):

        dataset = np.concatenate((X, y[:,None]), axis=1)  
        self.root = self.build_tree(dataset)
        # Create the tree graph
        self.tree_graph = self.create_tree_graph(self.root) 

    def predict(self, X):
        
        # Create an empty list to store the predictions
        predictions = []
        # For each instance in X, make a prediction by traversing the tree
        for x in X:
            prediction = self._predict(x, self.root)
            predictions.append(prediction)

        np.array(predictions)
        return predictions
    
    def _predict(self, x, node):

        # If the node has value i.e it's a leaf node extract it's value
        if node.value != None: 
            return node.value
        else:
            # If it's not a leaf we'll get its feature and traverse through the tree accordingly
            feature = x[node.feature]
            if feature <= node.threshold:
                return self._predict(x, node.left)
            else:
                return self._predict(x, node.right)
            
    def create_tree_graph(self, node, graph=None):

        if graph is None: # graph is the graph object.
            graph = Digraph(format='png')  # You can choose the desired format

        if node.value is not None:
            # If it's a leaf node, add the label
            graph.node(str(id(node)), label=str(node.value))
        else:
            # If it's an internal node, add the split information
            graph.node(str(id(node)), label=f"Feature: {node.feature}\nThreshold: {node.threshold}")
            graph = self.create_tree_graph(node.left, graph)
            graph = self.create_tree_graph(node.right, graph)
            graph.edge(str(id(node)), str(id(node.left)))
            graph.edge(str(id(node)), str(id(node.right)))

        return graph

    def visualize_tree(self, file_name='decision_tree'):
    # Save the plot

        self.tree_graph.render(filename=file_name, cleanup=True, format='png')

################################## LOGISTIC ##################################
class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
    
    def fit(self, X, y):
        self.X_train = self._add_intercept(X)
        y_for_logistic = np.array([0 if label == -1 else label for label in y]) # transform the -1 in 0 to train the logistic
        self.y_train = y_for_logistic
        self.weights = np.zeros(self.X_train.shape[1])
        
        for t in range(1,len(self.X_train)+1):
            scores = np.dot(self.X_train, self.weights)
            self.predictions = self._sigmoid(scores) # Mine
            # self.predictions = self._sigmoid(scores*self.y_train) # Cesa Bianchi

            # Gradient of logistic loss
            # Loss(y, y_hat) = -[y * log(y_hat) + (1-y) * log((1 - y_hat))] logarithmic or negative likelihood
            # Loss(y, y_hat) = log_2(1 + exp(-y*y_hat)) logistic
            gradient = np.dot(self.X_train.T, (self.predictions - self.y_train)) / self.y_train.shape[0] # Mine
            # gradient = ( -(np.dot((self.X_train.T*self.y_train), self.predictions) / np.log(2)) )  / self.y_train.shape[0] # Cesa Bianchi
            
            self.weights -= self.learning_rate/np.sqrt(t) * gradient
    
    def predict(self, X):
        X_with_intercept = self._add_intercept(X)
        scores = np.dot(X_with_intercept, self.weights)
        predicted_probabilities = self._sigmoid(scores)
        predicted_labels = (predicted_probabilities >= 0.5).astype(int) # boolean vector cast as integers 1 or 0
        binary_predicted_labels = np.array([-1 if label == 0 else label for label in predicted_labels]) # transform the 0 in -1 to return the binary predictions y_hat = {-1,1}
        return binary_predicted_labels
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.hstack((intercept, X))
    
################################## SVM ##################################
class SVM:

    def __init__(self, Lambda = 1.0):
        self.Lambda = Lambda
        self.w = 0
    
    def fit(self, X, y, learning_rate=0.001, epochs=1000):
        X = self._add_intercept(X) # constant term added because SVM is a linear predictor
        number_of_features = X.shape[1] # it includes the bias
        number_of_samples = X.shape[0]

        # Creating indices from 0 to number_of_samples - 1 to randomly draw from X
        ids = np.arange(number_of_samples)
        np.random.seed(42)
        np.random.shuffle(ids) # randomly shuffles the indices inplace in order to be drawn for each Pegasos update

        # Initialize the weights (w) by setting them to zero
        w = np.zeros((1, number_of_features)) # (1x10) row vector
        weights = []

        # Stochastic Gradient Descent (Pegasos)
        for t,i in zip(range(1, epochs+1), ids): # +1 added in order to avoid a division by zero when computing the learning rate η_t
            # Draw one from the randomly shuffled examples (X[i], y[i])
            # If the point is correctly classified don't make an update
            ywx = y[i] * np.dot(X[i], w.T) # (1x1) * (1x10) * (10x1)
            hinge_indicator = 0 if ywx >= 1 else 1 # I{h_t(w_t) > 0}
            # Gradient of the SVM objective: (λ/2)*||w||^2 + h_t(w_t)
            gradw = -y[i]*X[i]*hinge_indicator + self.Lambda*w # = ∇l_t(w_t) =  -y_t*x_t*I{h_t(w_t) > 0} + λw_t

            # Updating weights vector
            w = w - learning_rate/np.sqrt(t) * gradw
            # Append the new weights vector to the list
            weights.append(w)
        
        w_bar = sum(weights)/len(weights) # w_bar = (1/T)(w_1 + ... +  w_T)
        self.w = w_bar[0][1:] # extract the features coefficients \\ self.w is an array of one element: the weights array
        self.bias = w_bar[0][0] # extract the bias term           \\ self.w is an array of one element: the weights array
    
    def predict(self, X):
        X = self._add_intercept(X) # constant term added because SVM is a linear predictor
        weights_and_bias = np.concatenate((np.array([self.bias]), self.w))
        prediction = np.dot(X, weights_and_bias)
        return np.sign(prediction)
        
    def _add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.hstack((intercept, X))