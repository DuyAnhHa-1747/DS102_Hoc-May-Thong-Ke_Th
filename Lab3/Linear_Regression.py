import numpy as np 

class LinearRegression:
    def __init__(self):
        self._W = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        X: (N, D)
        y: (N, )
        """
        N, D = X.shape
        self._W = np.zeros((D, ))
        cov = X.T @ X  # (D, D)
        inv_cov = np.linalg.inv(cov)  # (D, D)
        self._W = inv_cov @ X.T @ y  # (D, )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        X: (N, D)
        """
        y_hat = X @ self._W  # (N, )

        return y_hat

    def evalute(self, y: np.ndarray, y_hat: np.ndarray):
        rmse = ((y - y_hat)**2).mean()**0.5    
        return rmse