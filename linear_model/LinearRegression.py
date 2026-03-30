"""
Things : - 

# it is simple linear regression
y = mx + b

y = prediction
m = weight
x = value
b = bias

loss = prediction - real_val
dw = gradient descent of weight
db = gradient descenf of bias



"""

import numpy as np

class SimpleLinearRegression():


    def __init__(self):

        """
        Simple Linear Regression
        ------------------------

        A Simple linear Regression for 1D arrays 

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

        >>> from LinearRegression import SimpleLinearRegression
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
            X : np.array,
            y : np.array, 
            learning_rate : float = 0.01,
            weights : float = 0,
            bias : float = 0,

            ):
        """

        Input Param*
        __________
        X = Data to Train 1D array, Dtype = np.array  

        Y = True value a.k.a. original prediction 1D array, Dtype = np.array

        epochs = loop to update weight and bias, Dtype = int and default = 100

        learning_rate = how fast weights should update, Dtype = float, Default = 0.01  

        weights = enter custom weight | optional  
        bias = enter custom bias | optional  
        -----------------

        Change the learning_rate if output or weights are contains 'e' e.g -1.8038873e+163
        """


        # Avoiding List or 2D array Edge Case
        X = np.array(X).reshape(-1)
        y = np.array(y).reshape(-1)


        n = len(X) # Length of X

        db = dw = 0
        for _ in range(15000):
            pred : np.array = (X * weights) + bias
            loss : np.array= pred - y

            if sum(loss) <= 0.2 and sum(loss) >= 0:
                print(f"Job Done with {sum(loss)} Loss")
                break

            db = sum(loss) * (2/n)
            dw = sum(loss * X) * (2/n)

            weights -= learning_rate * dw
            bias -= learning_rate * db
        
        self.weights = weights
        self.bias = bias

    def get_weight_bias(self) -> np.array:
        """Input = None, 
        O/P Type np.array
        
        Output Format
        arr[0] = weight
        arr[1] = bias
        arr[0] = arr[1] = float64 Dtype
        """

        return np.array([self.weights, self.bias])
    
    def predict(self, new_data : np.array) -> np.array:
        """
        Input Format = 1D np.array
        Output Format = 1d np.array
        """
        if len(new_data) == 0:
            raise ValueError("Got Empty Array")
        
        prediction = []
        for items in new_data:
            pred = items * self.weights + self.bias
            prediction.append(pred)
        
        return np.array(prediction).round(2)


if __name__ == "__main__":
    Model = SimpleLinearRegression()
    X = [10, 20, 30, 40, 50, 60]
    y = [1, 2, 3, 4, 5, 6]
    Model.fit(X, y, epochs=1000, learning_rate=0.00001)
    print(Model.get_weight_bias())

    print(Model.predict(np.array([70, 80])))