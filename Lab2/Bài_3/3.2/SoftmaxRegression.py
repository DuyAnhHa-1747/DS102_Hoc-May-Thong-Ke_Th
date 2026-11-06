import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

class SoftmaxRegression:
    def __init__(self, learning_rate: float = 0.1, epochs: int = 1000, num_classes: int = 3):
        self.num_classes = num_classes
        self.model = SklearnLogisticRegression(
            solver='lbfgs',
            max_iter=epochs,
            random_state=42
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)