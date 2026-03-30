import numpy as np

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.min_v = None
        self.max_v = None
        self.a, self.b = feature_range

    def fit(self, X):
        X = np.array(X)

        # if 1D → convert to 2D column
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.min_v = np.min(X, axis=0)
        self.max_v = np.max(X, axis=0)

    def transform(self, X):
        X = np.array(X)
        is_1d = False

        if X.ndim == 1:
            X = X.reshape(-1, 1)
            is_1d = True

        denom = self.max_v - self.min_v
        denom = np.where(denom == 0, 1, denom)

        X_scaled = self.a + (X - self.min_v) * (self.b - self.a) / denom

        # return back to 1D if input was 1D
        if is_1d:
            return X_scaled.reshape(-1)

        return X_scaled

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        X = np.array(X)
        is_1d = False

        if X.ndim == 1:
            X = X.reshape(-1, 1)
            is_1d = True

        X_orig = (X - self.a) * (self.max_v - self.min_v) / (self.b - self.a) + self.min_v

        if is_1d:
            return X_orig.reshape(-1)

        return X_orig

if __name__ == "__main__":
    Scaler = MinMaxScaler()
    from sklearn.preprocessing import  MinMaxScaler