"""
Things : - 

# it is linear regression
y = m1x1 + m1x2 + m3x3 + ... + MnXn + b

y = prediction
m = weight
x = value
b = bias

loss = prediction - real_val
dw = gradient descent of weight
db = gradient descenf of bias

It uses Gradients so, Use `StandardScaler` or `MinMaxScaler` for better result  

Scalers...
>>> from rslearn.preprocessing import StandardScaler, MinMaxScaler
Read READNE.md or Documentation for More Information about their Functions
"""

import numpy as np
from rslearn.metrics import mse

class LinearRegression():


    def __init__(self, regulization=None, alpha : float = 0.1, l1_ratio=0.5):

        """
        Linear Regression
        ------------------------

        linear Regression for 1D and 2D metrics arrays  using gradient descents and regulization  
        use Scalers like MinMaxScaler or StandardScaler before fitting for haldle large value  

        Example
        --------
        regulization: regulizing option to avoid overfitting  
            options:  `l1` for Lasso  
                      `l2` for Ridge  
                      `elastic_net` for elastic_net
            
            Default: None, For No regulization.
        
        alpha: alpha value for Ridge, Lasso, ElasticNet  
            Default: 0.1  
        
        l1_ratio: Lasso Ratio for Better ElasticNet Gradient and MSE  
            Default: 0.5  
        
        Functions
        ---------
        fit()
            Function for Train Model | use MinMaxScaler for good Computation and Prediction  
            Parameters - given in Function
        
        get_weight_bias()
            Returns Selected weight and Bias for minimum loss  
        
        predict()
            Prediction generator from Model  
        
        
        Example
        -------

        >>> from rslearn.linear_model import LinearRegression
        >>> Model = LinearRegression()
        >>> X = np.array([10, 20, 30]) # List also works.
        >>> y = np.array([5, 10, 15])
        >>> Model.fit(X, y) # You can change learning_rate too
        >>> print(f"Weight & Bias: {Model.get_weight_bias()}")
        >>> prediction = Model.predict(np.array([40, 50]))
        >>> print(f"Prediction: {prediction}")
            

        
        """

        self.weights = None
        self.bias = None

        valid_params = {"l1", "l2", "elastic_net", None}
        if regulization not in valid_params:
            raise ValueError(f"regulization parameter is not supported, supported Parameters {valid_params}")
        
        self.caclucate_error = self._regulizing_linear_helper(regulization=regulization, alpha=alpha, l1_ratio=l1_ratio)
            
    


    def fit(self,
            X ,
            y , 
            weights= None,
            bias = None,
            learning_rate : float = 0.01,
            min_loss : float = 0.2,
            max_itr : int = 18000
            ):
        """

        Input Param*
        __________
        X = Data to Train 1D or 2D array, Dtype = np.array   

        Y = True value a.k.a. original prediction 1D or 2D array, Dtype = np.array  

        max_itr = loop to update weight and bias, Dtype = int and default = 18000  | No Input need  

        learning_rate = how fast weights should update, Dtype = float, Default = 0.01  

        weights = enter custom weight |  optional  

        bias = enter custom bias |  optional  

        min_loss = minimum loss where to stop the loop Default = 0.2 and its almost best for gradient descent  
        -----------------

        Change the `learning_rate` or Use `Scalers` if output or weights contains 'e' e.g -1.8038873e+163
        """

        

        X = np.array(X)
        y = np.array(y).reshape(-1)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_feature = X.shape

        np.random.seed(7)
        if weights is None:
            weights = np.random.uniform(0.2, 3, n_feature)
        
        if bias is None:
            bias = 0

        iteration  = 0

        while iteration < max_itr:
            pred = np.dot(X, weights) + bias # prediction

            mse_error = self.caclucate_error.get_error(y_true=y, y_pred=pred, weights=weights)

            if mse_error <= min_loss:
                print(f"Model Succesfully Fitted at #{iteration} iteration")
                break

            loss = pred - y # Loss for Gradients
            dw = (2/n_samples) * np.dot(X.T, loss) + self.caclucate_error.get_weight_gradient(weights=weights)
            db = (2/n_samples) * np.sum(loss)

            weights -= learning_rate * dw
            bias -= learning_rate * db

            if np.isnan(weights).any() or np.isnan(bias):
                print("NaN detected, stopping training, Use Scalers to avoid it")
                break

            iteration += 1


        
        self.weights = weights
        self.bias = bias

    def get_weight_bias(self) -> np.array:
        """Input = None, 
        O/P - (np.array, float64)
        >>> weights, bias = Model.get_weight_bias()
        """

        return (self.weights, self.bias)
    
    def predict(self, new_data : np.array) -> np.array:
        """
        Input Format = 1D or 2D np.array
        Output Format = 1D np.array
        """
        if len(new_data) == 0:
            raise ValueError("Got Empty Array")
        
        new_data = np.array(new_data)

        if new_data.ndim == 1:
            new_data = new_data.reshape(-1, 1)

        return (np.dot(new_data, self.weights) + self.bias).round(2)

    class _regulizing_linear_helper:
        def __init__(self, alpha=0.1, regulization=None, l1_ratio = 0.5):
            self.alpha = alpha
            self.regulization = regulization
            self.l1_ratio = l1_ratio
        
        def get_error(self, y_true, y_pred, weights):
            mse_error = mse(y_true, y_pred)
            if self.regulization is None:
                return mse_error
            
            if self.regulization == "l1":
                reg = self.alpha * np.sum(np.abs(weights))
                return  mse_error + reg
            
            if self.regulization == "l2":
                reg = self.alpha * np.sum(np.square(weights))
                return mse_error + reg
            
            if self.regulization == "elastic_net":
                l1 = self.alpha * self.l1_ratio
                l2 = self.alpha * (1 - self.l1_ratio)

                reg = l1 * np.sum(np.abs(weights)) + l2 * np.sum(np.square(weights))
                return mse_error + reg
        
        def get_weight_gradient(self, weights):
            if self.regulization == "l1":
                return self.alpha * np.sign(weights)
            
            if self.regulization == "l2":
                return 2 * self.alpha * weights
            
            if self.regulization == "elastic_net":
                l1 = self.alpha * self.l1_ratio
                l2 = self.alpha * (1 - self.l1_ratio)

                return l1 * np.sign(weights) + 2 * l2 * weights

            return 0





    

if __name__ == "__main__":
    Model = LinearRegression()