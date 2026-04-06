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

import numpy as np

class StandardScaler:
    def __init__(self) -> None:
        """
        StandardScaler

        A feature scaling utility that standardizes data by removing the mean
        and scaling to unit variance.

        This transformation is commonly used to normalize features so that they
        contribute equally to model training, especially for gradient-based
        algorithms.

        Methods
        -------
        fit(data: np.ndarray) -> None
            Computes the mean and standard deviation of the input data.

        transform(data: np.ndarray) -> np.ndarray
            Transforms the data using the previously computed mean and standard deviation.

        fit_transform(data: np.ndarray) -> np.ndarray
            Fits the scaler to the data, then returns the transformed result.
        
        Examples
        --------
        >>> from preprocessing.Scaler import StandardScaler
        >>> Scaler = StandardScaler()
        >>> X_Scaled = Scaler.fit(np.array([10, 20, 30])) # List Also works, np.array prefered
        >>> X_original = Scaler.inverse_transform(X_Scaled)
        """
        self.mean = None
        self.std = None

    def fit(self, data: np.ndarray) -> None:
        """
        Compute the mean and standard deviation of the input data.

        Parameters
        ----------
        data : np.ndarray
            Input array of shape (n_samples,) or (n_samples, n_features).

        Returns
        -------
        None
        """

        X = np.array(data)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std = np.where(self.std == 0, 1, self.std)


    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Standardize the input data using the computed statistics.

        Parameters
        ----------
        data : np.ndarray
            Input array of shape (n_samples,) or (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Scaled data with zero mean and unit variance.
        """

        if self.std is None:
            raise ValueError("Scaler has not been fitted yet.")

        X = np.array(data)
        is_1D = False

        if X.ndim == 1:
            X = X.reshape(-1, 1)
            is_1D = True
        
        denom = X - self.mean
        X_scaled = denom / self.std

        if is_1D:
            return X_scaled.reshape(-1)

        return X_scaled


    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit to the data, then transform it.

        Parameters
        ----------
        data : np.ndarray
            Input array of shape (n_samples,) or (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Scaled data with zero mean and unit variance.
        """
        self.fit(data)
        return self.transform(data)
    
    def inverse_transform(self, data_scaled : np.array) -> np.array:
        
        """
        Revert standardized data back to original scale.

        Parameters
        ----------
        data : np.ndarray
            Scaled data of shape (n_samples,) or (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Original data before scaling.
        """

        if self.std is None:
            raise ValueError("Scaler has not been fitted yet!")

        X = np.array(data_scaled)
        is_1D = False

        if X.ndim == 1:
            is_1D = True
            X = X.reshape(-1, 1)
        
        X_original = X * self.std + self.mean
        if is_1D:
            return X_original.reshape(-1)
        
        return X_original


if __name__ == "__main__":
    Scaler = MinMaxScaler()
    