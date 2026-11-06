import numpy as np
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression

class LogisticRegression:
    def __init__(self, learning_rate: float = 0.1, epochs: int = 1000, l2_penalty: float = 0.1, verbose: bool = False):
        self.epochs = epochs
        self.l2_penalty = l2_penalty
        c_val = 1.0 / l2_penalty if l2_penalty > 0 else 1.0
        self.model = SklearnLogisticRegression(
            C=c_val,
            max_iter=epochs,
            solver='lbfgs',
            random_state=42
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        if threshold == 0.5:
            return self.model.predict(X)
        probs = self.model.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)