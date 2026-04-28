# linear_models 
Collection of most of commonly used linear_model like Linear Regression Logistic Regression
Scaler is preferd for better result

## contains Models
- Linear Regression
- Logistic Regression
- Ridge & Lasso & ElasticNet Regulization (l1, l2 and ElasticNet)

## How to use
its pretty Simple Just import define and fit and then predict like sklearn like  

### `Linear Regression`
**new: Added Regulization Support For it**

#### `regulization` options
| Option            | Description                             |
| ----------------- | --------------------------------------- |
| `l1`              | Lasso Regulization                      |
| `l2`              | Ridge Regulization                      |
| `ElasticNet`      | ElasticNet Regulization                 |

***Read Doc Strings For More Prameter Knowledge***

```python
from rslearn.linear_model import LinearRegression
Model = LinearRegression(regulization="l1")
```

### `Logistic Regression`
StandardScaler or MinMaxScaler is preferd in Multi class classification
``` python
from rslearn.linear_model import LogisticRegression
Model = LogisticRegression()
```
checkout preprocessing/README.md for Scalers detail

Thats It! 

### `Ridge`, `Lasso`, `ElasticNet`
Regulizations For avoid overfitting

`New parameter`:  
`Scale=True` Automaticly Scale Data before sending to LinearRegression,  
Use `Scalers`, e.g `StandardScaler`, `MinMaxScaler` for better performance

``` python
from rslearn.linear_model import Lasso, Ridge, ElasticNet
```


### Documentation is coming! Explained All Parameters In that.
### `More Coming Soon`

#### Maden with ♥