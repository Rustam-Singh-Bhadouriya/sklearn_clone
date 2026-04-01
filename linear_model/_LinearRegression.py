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


"""

import numpy as np

class LinearRegression():


    def __init__(self):

        """
        Linear Regression
        ------------------------

        A Simple linear Regression for 1D and 2D metrics arrays  using gradient descents  
        use Scalers like MinMaxScaler or StandardScaler before fitting for haldle large value  

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

        >>> from LinearRegression import LinearRegression
        >>> Model = SimpleLinearRegression()
        >>> X = np.array([10, 20, 30]) # List also works.
        >>> y = np.array([5, 10, 15])
        >>> Model.fit(X, y) # You can change learning_rate too
        >>> print(f"Weight & Bias: {Model.get_weight_bias()}")
        >>> prediction = Model.predict(np.array([40, 50]))
        >>> print(f"Prediction: {prediction}")
            

        
        """

        self.weights = None
        self.bias = None


    def fit(self,
            X ,
            y , 
            weights: np.array = np.array([0, 0]),
            bias : float = 0,
            learning_rate : float = 0.01,
            min_loss : float = 0.2,
            max_itr : int = 18000
            ):
        """

        Input Param*
        __________
        X = Data to Train 1D array, Dtype = np.array   

        Y = True value a.k.a. original prediction 1D array, Dtype = np.array  

        max_itr = loop to update weight and bias, Dtype = int and default = 18000  | No Input need  

        learning_rate = how fast weights should update, Dtype = float, Default = 0.01  

        weights = enter custom weight |  optional  

        bias = enter custom bias |  optional  

        min_loss = minimum loss where to stop the loop Default = 0.2 and its almost best for gradient descent  
        -----------------

        Change the learning_rate if output or weights are contains 'e' e.g -1.8038873e+163
        """

        

        X = np.array(X)
        y = np.array(y).reshape(-1)

        if X.ndim == 1:
            X = X.reshape(-1, 1)
        
        n_samples, n_feature = X.shape

        np.random.seed(7)
        weights = np.random.uniform(0.2, 3, n_feature)
        bias = 0

        iteration  = 0

        while iteration < max_itr:
            pred = np.dot(X, weights) + bias # prediction

            mse_error = np.mean((pred - y) ** 2)

            if abs(mse_error) <= min_loss:
                print(f"Model Succesfully Fitted at #{iteration} iteration")
                break

            loss = pred - y # Loss for Gradients
            dw = (2/n_samples) * np.dot(X.T, loss)
            db = (2/n_samples) * np.sum(loss)

            weights -= learning_rate * dw
            bias -= learning_rate * db

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



if __name__ == "__main__":
    Model = LinearRegression()