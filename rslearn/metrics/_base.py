import numpy as np


def check_multioutput(parameter) -> None:
    valid_params = {
        "weighted",
        "uniform_average",
        "raw_values"
        }

    if parameter not in valid_params:
        raise ValueError(
            f"Got Invalid Parameter, {parameter}. But supported {valid_params} only."
            )

    return

def convert_array(arr1, arr2) -> np.ndarray:
    arr1 = np.asarray(arr1, dtype=float)
    arr2 = np.asarray(arr2, dtype=float)

    return arr1, arr2

def convert1D(*args) -> np.array:
    """
    Parameters
    ----------
    y_true,  
    y_pred,  
    any, any  

    Returns
    -------
    list of Flat Arrays by  
    X, y = convert1D(X, y)   
    X, y, z = convert1D(X, y, z)  

    """
    arrays = []
    for items in args:
        items, _ = convert_array(items, [1012, 1203])
        arrays.append(np.ravel(items))
    
    return tuple(np.array(arrays))

def shape_checker(arr1, arr2, output_mode=True): 
    arr1, arr2 - convert_array(arr1, arr2)

    # if arr1, arr2 are coming from classification y_true, y_pred  
    # to avoid multioutput shape error like  
    # Mendontory for Classification Task  
    """

    From Regression Output mainly `/metrics/_regression.py`
    y_true = np.array(
        [
            [10, 20],
            [30, 25]
        ]
    )

    y_pred = np.array(
        [
            [10.8, 19.9],
            [30, 23.7]
        ]
    )

    From Classification Output mainly `/linear_model/_Logistic` & `/metrics/_classification.py`


    y_true = np.array(
        [
            [1, 1],
            [0, 1]
        ]
    )

    y_pred = np.array(
        [
            [1, 0],
            [0, 1]
        ]
    )

    """
    if output_mode:
        if arr1.shape != arr2.shape:
            raise ValueError(f"Shape mismatch: {arr1.shape} vs {arr2.shape}")

    # if arr1, arr2 are Coming from Regression X, y
    
    if len(arr1) != len(arr2): # Assume as X, y
        raise ValueError(f"Shape mismatch: {len(arr1)} vs {len(arr2)}")

def dim_validator(arr1):
    """
    Dimensional Matcher For `/metrics/*`  
    """
    y_true, _ = convert_array(arr1, [10]) # Converting to Array

    if y_true.ndim == 1: # Checking direct Dimensional for [1, 2, 3]
        return True
    
    if y_true.shape[1] == 1:# Checking Indirect 1D array like [[1], [2], [3]]
        return True

    return False

def multi_output_selector(multi_output_param, scores, weights):
    check_multioutput(multi_output_param) # Safty Check if Internel Errors Happens

    match multi_output_param:
        case "raw_values":
            return scores
        
        case "uniform_average":
            return np.mean(scores)
        
        case "weighted":
            if weights is None:
                raise ValueError("weights must be provided for weighted averaging")
            
            weights = np.asarray(weights, dtype=float)

            if weights.shape[0] != len(scores):
                raise ValueError("weights length must match number of outputs")
            
            return np.average(scores, weights=weights)
    


        

if __name__ == "__main__":
    X = [[12], [11], [10]]
    y = [[13], [11], [8]]
    X, y= convert1D(X, y)
    print(type(X), y)

