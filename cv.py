from sklearn.model_selection import KFold  # Use sklearn's KFold for cross-validation
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

class KFold_cv:
    def __init__(self, model, num_folds=5):
        self.model = model
        self.num_folds = num_folds

    def cv_metrics(self, X, y, metrics=["accuracy","precision","recall","roc_auc", "f1"]):
        # Type check and possible correction for str input
        if isinstance(metrics, str):
            metrics = [metrics]
            
        # Initialize a dictionary to store metrics
        metrics_dict = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "roc_auc": [],
            "f1": []
        }

        # Initialize KFold cross-validator
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=42)

        for train_index, test_index in kf.split(X): # creates train and test indices 5 times
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Train your model
            self.model.fit(X_train, y_train)

            # Make predictions (y_hat)
            predictions = self.model.predict(X_test)

            # Compute metrics
            accuracy = accuracy_score(y_test, predictions)
            precision = precision_score(y_test, predictions)
            recall = recall_score(y_test, predictions)
            roc_auc = roc_auc_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)

            metrics_dict["accuracy"].append(accuracy)
            metrics_dict["precision"].append(precision)
            metrics_dict["recall"].append(recall)
            metrics_dict["roc_auc"].append(roc_auc)
            metrics_dict["f1"].append(f1)

        # Calculate average metrics
        avg_metrics = {
            "accuracy": sum(metrics_dict["accuracy"]) / self.num_folds,
            "precision": sum(metrics_dict["precision"]) / self.num_folds,
            "recall": sum(metrics_dict["recall"]) / self.num_folds,
            "roc_auc": sum(metrics_dict["roc_auc"]) / self.num_folds,
            "f1": sum(metrics_dict["f1"]) / self.num_folds
        }   
        return [avg_metrics[metric] for metric in metrics]