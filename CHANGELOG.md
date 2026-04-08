# Changelog

All notable changes to this project will be documented in this file.

---

## [1.0.0] - 2026-04-07

### Added

* Linear Regression (Gradient Descent based)
* Logistic Regression (Binary & Multi-class)
* Ridge (L2 Regularization)
* Lasso (L1 Regularization)
* Elastic Net (L1 + L2)
* StandardScaler and MinMaxScaler
* Train-Test Split with stratify support
* Metrics:

  * MSE, MAE, RMSE, R² Score, Accuracy
* Support for single and multi-output tasks

### Improved

* Clean sklearn-like API design
* Modular folder structure
* Internal documentation for each module
* CHANGELOG.md
* Doc Strings
* Basic metrics problems

### Notes

* Models use Gradient Descent — feature scaling is highly recommended
* Regularized models include optional auto-scaling (`scale=True`)
