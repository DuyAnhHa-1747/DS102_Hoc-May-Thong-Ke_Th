import numpy as np

class LogisticRegression:
    # Hàm khởi tạo
    def __init__(self, learning_rate=0.05, epochs=500, verbose=False):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.verbose = verbose
        self.weights = None
        self.bias = None
        self.losses = []

    # Hàm đổi biến số sang xác xuất
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500) # tránh -z hoặc z quá lớn làm bị overflow
        return 1.0 / (1.0 + np.exp(-z))

    # Hàm mất mát 
    def compute_loss(self, y_true, y_pred):
        eps = 1e-15 # tránh log(0)
        y_pred = np.clip(y_pred, eps, 1.0 - eps)
        loss = -np.mean(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
        return loss

    def fit(self, X, y):
        N, d = X.shape
        self.weights = np.zeros(d)
        self.bias = 0.0
        self.losses = []

        for epoch in range(1, self.epochs + 1):
            # z = Xw + b
            z = np.dot(X, self.weights) + self.bias
            # chuyển đổi biến số
            y_pred = self._sigmoid(z)

            # Tính loss
            loss = self.compute_loss(y, y_pred)
            self.losses.append(loss)

            # Tính gradients 
            dw = (1.0 / N) * np.dot(X.T, (y_pred - y))
            db = (1.0 / N) * np.sum(y_pred - y)

            # Cập nhật tham số
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self 

    # Dự đoán xác suất
    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    # Phân loại dựa trên threshold = 0.5
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)