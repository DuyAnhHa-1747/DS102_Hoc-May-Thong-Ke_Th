import numpy as np

class SoftmaxRegression:
    # khởi tạo
    def __init__(self, learning_rate: float = 0.05, epochs: int = 500, num_classes: int = 3):
        self.lr = learning_rate
        self.epochs = epochs
        self.num_classes = num_classes
        self.weights = None
        self.bias = None
        self.losses = []

    # tính xác suất softmax
    def _softmax(self, Z: np.ndarray) -> np.ndarray:
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        m, n = X.shape
        self.weights = np.zeros((n, self.num_classes))
        self.bias = np.zeros((1, self.num_classes))
        self.losses = []

        # mã hoá nhãn (One-Hot Encoding)
        y_encoded = np.zeros((m, self.num_classes))
        y_encoded[np.arange(m), y] = 1.0

        for epoch in range(1, self.epochs + 1):
            # tính tổng trọng số, Z = XW + b
            Z = np.dot(X, self.weights) + self.bias 
            
            probabilities = self._softmax(Z)

            # Loss (Categorical Cross-Entropy)
            eps = 1e-15
            prob_clipped = np.clip(probabilities, eps, 1.0 - eps)
            loss = -np.mean(np.sum(y_encoded * np.log(prob_clipped), axis=1))
            self.losses.append(loss)

            # Gradient Descent
            dZ = probabilities - y_encoded
            dW = (1.0 / m) * np.dot(X.T, dZ)
            dB = (1.0 / m) * np.sum(dZ, axis=0, keepdims=True)

            # Cập nhật trọng số
            self.weights -= self.lr * dW
            self.bias -= self.lr * dB

        return self

    # dự đoán xác suất
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = np.dot(X, self.weights) + self.bias
        return self._softmax(Z)

    # phân loại 
    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)