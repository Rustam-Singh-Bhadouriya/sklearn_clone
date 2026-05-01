# 🪈 rslearn.Pipeline

beginner-friendly pipeline that removes the need for manual preprocessing, splitting, and evaluation

Includes - 
* **`pipeline`**

---
## 🎯 Why rslearn.Pipeline?

- Beginners often struggle with multiple steps (scaling, splitting, evaluation)
- This pipeline automates everything into a single workflow
- Focus on learning ML concepts instead of handling boilerplate code

---

## 🚀 Features
* **Automate Data `Scaling & Training`**
* **`Validation_split` Automate Data Splitter And Tester For test Data**
* **Inbuilt Analysis Tools Contains**
    * For Classification
        * `accuracy_score`
        * `recall`
        * `f1_score`
        * `precision`  
        Calculations  
    * For Regression
        * `r2_score`
        * `Mean Squared Error`
        * `Mean Absolute Error`
        * `Root Mean Squared Error`
* Support Regulization Class Support
    * `Ridge`
    * `Lasso`
    * `ElasticNet`

## 📐 Parameters

### 🔹 `params`
Parameters Which Contains `Model` & `Scaler` to Train and Evaluate

**Dtype = Dict**  


**Options**  
| key name | Description | options |  
|----------|-------------|---------|  
| `model`  | Model To Train| LinearRegression etc.|  
| `scaler` | To Scale Data | StandardScaler or MinMaxScaler| 


### 🔹 Remain Parameters
| Parameter Name | Description | Option | Default | Dtype | Recommended | 
|----------------|-------------|--------|---------|-------|-----------|  
| `validation_split` | Auto Splitter for Auto Test accuracy| True/False| False | **Bool** | `True`
| `split_params` | Splitting Configurations| test_size, random_state, stratify| given in class| **Dict** | `change stratify if Needed`


## 🛠️ Functions
| Function Name | Description | Parameter |  
|---------------|-------------|-----------|
| `fit` | To Train The Model | **X** - NxM metrics of Features, **y** - Correct Output For X Metrics|
| `predict` | To Predict From Pipeline | **new_data** - New NxM metrics of Float |
| `analysis` | Function to print All metrics Evaluation E.g `accuracy_score`, `recall`, etc| **y_pred** - prediction from pipeline & **y_true** - Correct Values|

## 🧑‍💻 Code

***Using `Validation_split` = `False`***

``` python
from rslearn.preprocessing import StandardScaler
from rslearn.linear_model import LinearRegression

line = pipeline(
    params={
        "model": LinearRegression(regulization="l1"),
        "scaler": StandardScaler()
    },
    validation_split=False,
    )

    X = [
        [10, 20], 
        [40, 50], 
        [15, 15], 
        [40, 35], 
        [12, 15], 
        [25, 15], 
        [13, 12], 
        [15, 10], 
        [15, 15]
    ]
    y = [
        [30],
        [90],
        [30],
        [75],
        [27],
        [40],
        [25],
        [25],
        [30]
    ]


    line.fit(X=X, y=y)
    pred = line.predict([[34, 30], [10, 23], [10, 25]])
    print(pred)
    line.analysis(y_pred=pred, y_true=[[64], [33], [35]])
```

***Using `Validation_split` = `True`***

``` python
from rslearn.preprocessing import StandardScaler
from rslearn.linear_model import LinearRegression

line = pipeline(
    params={
        "model": LinearRegression(regulization="l1"),
        "scaler": StandardScaler()
    },
    validation_split=True,
    )

    X = [
        [10, 20], 
        [40, 50], 
        [15, 15], 
        [40, 35], 
        [12, 15], 
        [25, 15], 
        [13, 12], 
        [15, 10], 
        [15, 15]
    ]
    y = [
        [30],
        [90],
        [30],
        [75],
        [27],
        [40],
        [25],
        [25],
        [30]
    ]


    line.fit(X=X, y=y)
```
Auto Split & Test Evaluation And It will print Automatically, You Dont need to Call The Analysis Function For Given Data And Also Dont needed Any train_test_split

**What It will do Automatically**
* split the data into train & test data
* fit the model
* predicts for test data
* Evaluate Metrics Stuff & prints it

You will only need `analysis` Function When You predict for Another Data out of The Data you direct Given

### 🔺 Profit From `Validation_split=True`
* You Dont Need to Manually split the data into train & test
* You Have Full Control too, like `test_size` & `stratify`
* You Dont Need to use `accuracy_score` etc Manually, It will prints All Analysis For you


#### You will only need `analysis` Function When You predict for Another Data out of The Data you direct Given