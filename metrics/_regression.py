import numpy as np

"""
`_regression.py`, By: Rustam Singh

This File Contains All Important and most used error comupation algorithams,  
like  
- r2_score  
- MAE  
- MSE  
- RMSE  

All Methords
-------------
- r2_score  
- MAE (Coming Soon)  
- MSE (Coming Soon)  
- RMSE (Coming Soon)  
"""


def r2_score(y_true, y_pred, multi_output="uniform_average", weights=None):
    """
    Compute the coefficient of determination (R² score).

    The R² score represents the proportion of variance in the dependent
    variable that is predictable from the independent variables.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Estimated target values.

    multi_output : {"uniform_average", "weighted", "raw_values"}, default="uniform_average"
        Defines how to aggregate scores for multi-output data:
        
        - "uniform_average" : Average scores across all outputs (default)
        - "weighted"        : Weighted average using `weights`
        - "raw_values"      : Return scores for each output separately

    weights : array-like of shape (n_outputs,), optional
        Weights used when `multi_output="weighted"`. Higher values indicate
        greater importance of the corresponding output.

    Returns
    -------
    float or np.ndarray of shape (n_outputs,)
        R² score. Returns a single float for 1D targets or aggregated output,
        or an array of scores when `multi_output="raw_values"`.

    Examples
    --------
    Basic usage with 1D targets:

    >>> from sklearn_clone.metrics import r2_score
    >>> y_true = [3, -0.5, 2, 7]
    >>> y_pred = [2.5, 0.0, 2, 8]
    >>> r2_score(y_true, y_pred)

    Multi-output regression (automatic averaging):

    >>> y_true = [[3, 10], [-0.5, 20], [2, 30], [7, 40]]
    >>> y_pred = [[2.5, 12], [0.0, 18], [2, 33], [8, 39]]
    >>> r2_score(y_true, y_pred)

    Weighted multi-output:

    >>> r2_score(y_true, y_pred, multi_output="weighted", weights=[0.6, 0.4])
    """

    valid_options = {"uniform_average", "weighted", "raw_values"}
    if multi_output not in valid_options:
        raise ValueError(
            f"Invalid multi_output={multi_output}. Supported: {valid_options}"
        )

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")

    if y_true.size < 2:
        raise ValueError("At least 2 samples are required")

    # Handle 1D case
    if y_true.ndim == 1:
        return r2_score_1d(y_true, y_pred)

    # Multi-output case
    r2_scores = r2_score_helper_2D(y_true, y_pred, score_only=True)

    if multi_output == "raw_values":
        return r2_score

    if multi_output == "uniform_average":
        return np.mean(r2_scores)

    if multi_output == "weighted":
        if weights is None:
            raise ValueError("weights must be provided for weighted averaging")

        weights = np.asarray(weights)

        if weights.shape[0] != len(r2_scores):
            raise ValueError("weights length must match number of outputs")

        return np.average(r2_scores, weights=weights)


def r2_score_1d(y_true, y_pred):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    y_mean = np.mean(y_true)

    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - y_mean) ** 2)

    if denominator == 0:
        return 0.0

    return 1 - (numerator / denominator)

def r2_score_helper_2D(y_true, y_pred, weight=None, score_only=False):
    n_col = y_true.shape[1]
    
    r2_scores = []
    for cols in range(n_col):
        selected_true = y_true[:, cols]
        selected_pred = y_pred[:, cols]

        score = r2_score_1d(selected_true, selected_pred)
        r2_scores.append(score)
    
    if score_only:
        return np.array(r2_scores)

    elif weight is None:
        return np.mean(r2_scores)
    
    else:
        if len(weight) != n_col:
            raise ValueError("Invalid weights given")
        
        return np.average(r2_scores, weights=weight)

def mse(
    y_true,
    y_pred,
    multi_output="uniform_average",
    weights=None
):
    
    """
    Mean Squared Error  

    The Mean Squared Error measures the average squared difference between
    the true target values and the predicted values. It penalizes larger
    errors more heavily due to the squaring operation, making it sensitive
    to outliers.

    Parameter
    ---------
    y_true : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Predicted target values.
    
    multi_output : {"uniform_average", "weighted", "raw_values"}, default="uniform_average"
        Defines how to aggregate error for multi-output data:
        
        - "uniform_average" : Average error across all outputs (default)
        - "weighted"        : Weighted average using `weights`
        - "raw_values"      : Return error for each output separately

    weights : array-like of shape (n_outputs,), optional
        Weights used when `multi_output="weighted"`. Higher values indicate
        greater importance of the corresponding output.

    Returns
    -------
    float or np.ndarray of shape (n_outputs,)
        mean squared error Returns a single float for 1D targets or aggregated output
        or an array of scores when `multi_output="raw_values"`.
    
    Raise
    -----
    ValueError: when array's Shape MisMatch or Invalid `multi_output` and Invalid `weights`
    

    Example
    -------
    >>> from sklearn_clone.metrics import mse # Importing
    >>> # 1D case
    >>> print(mse([1, 2, 3], [1, 2, 4]))  # expected: 0.333...

    >>> # multi-output
    >>> y_true = [[1, 2], [3, 4]]
    >>> y_pred = [[1, 3], [2, 5]]

    >>> print(mse(y_true, y_pred, multi_output="raw_values"))
    """
    
    valid_options = {"uniform_average", "weighted", "raw_values"}
    if multi_output not in valid_options:
        raise ValueError(
            f"Invalid multi_output={multi_output}. Supported: {valid_options}"
        )
    


    # Formula = 1/n sum((y_true - y_pred)**2)

    y_true = np.asarray(y_true, dtype=float) # Changing to np.array
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape Mismatch Error {(y_true.shape, y_pred.shape)}")
    
    if y_true.shape[1] == 1 or y_true.ndim == 1:
        return mse_helper_1d(y_true, y_pred)
    
    output_errors = mse_helper_2D(y_true, y_pred)

    if multi_output == "raw_values":
        return output_errors
    
    if multi_output == "uniform_average":
        return np.mean(output_errors)
    
    if multi_output == "weighted":
        if weights is None:
            raise ValueError("weights must be provided for weighted averaging")
        
        weights = np.asarray(weights)
        
        if weights.shape[0] != len(output_errors):
            raise ValueError(f"Invalid Weight Size got {weights.shape[0]}, needed {len(output_errors)}")
        
        return np.average(output_errors, weights=weights)
    


# MSE helper for 1 columns Simple Regression Ouput
def mse_helper_1d(y_true: np.array, y_pred: np.array):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)

    n = len(y_true)

    output_error = np.sum((y_true - y_pred)**2) / n

    return float(output_error)


# MSE helper for multioutput Regression Tasks
def mse_helper_2D(y_true: np.array , y_pred : np.array):
    return np.mean((y_true - y_pred) ** 2, axis=0) # vector Calculation
