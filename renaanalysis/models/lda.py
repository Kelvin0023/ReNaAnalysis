import numpy as np


class LDA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.linear_discriminants = None

    def fit(self, X, y):
        n_features = X.shape[1]
        class_labels = np.unique(y)

        X_mean_overall = np.mean(X, axis=0)
        std_within_class = np.zeros((n_features, n_features))
        std_betweem_classes = np.zeros((n_features, n_features))

        for c in class_labels:
            X_this_class = X[y == c]
            X_mean_this_class = np.mean(X_this_class, axis=0)
            # (4, n_c) * (n_c, 4) = (4,4) -> transpose
            std_within_class += (X_this_class - X_mean_this_class).T.dot((X_this_class - X_mean_this_class))

            # (4, 1) * (1, 4) = (4,4) -> reshape
            num_X_this_class = X_this_class.shape[0]
            mean_diff = (X_mean_this_class - X_mean_overall).reshape(n_features, 1)
            std_betweem_classes += num_X_this_class * (mean_diff).dot(mean_diff.T)