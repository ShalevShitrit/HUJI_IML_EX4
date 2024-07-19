from typing import NoReturn
import numpy as np
from base_estimator import BaseEstimator
from gradient_descent import GradientDescent
from modules import L1, L2, LogisticModule, RegularizedModule
from loss_functions import misclassification_error


class LogisticRegression(BaseEstimator):
    """
    Logistic Regression Classifier

    Attributes
    ----------
    solver_: GradientDescent, default=GradientDescent()
        Descent method solver to use for the logistic regression objective optimization

    penalty_: str, default="none"
        Type of regularization term to add to logistic regression objective. Supported values
        are "none", "l1", "l2"

    lam_: float, default=1
        Regularization parameter to be used in case `self.penalty_` is not "none"

    alpha_: float, default=0.5
        Threshold value by which to convert class probability to class value

    include_intercept_: bool, default=True
        Should fitted model include an intercept or not

    coefs_: ndarray of shape (n_features,) or (n_features+1,)
        Coefficients vector fitted by linear regression. To be set in
        `LogisticRegression.fit` function.
    """

    def __init__(self,
                 include_intercept: bool = True,
                 solver: GradientDescent = GradientDescent(),
                 penalty: str = "none",
                 lam: float = 1,
                 alpha: float = .5):
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        solver: GradientDescent, default=GradientDescent()
            Descent method solver to use for the logistic regression objective optimization

        penalty: str, default="none"
            Type of regularization term to add to logistic regression objective. Supported values
            are "none", "l1", "l2"

        lam: float, default=1
            Regularization parameter to be used in case `self.penalty_` is not "none"

        alpha: float, default=0.5
            Threshold value by which to convert class probability to class value

        include_intercept: bool, default=True
            Should fitted model include an intercept or not
        """
        super().__init__()
        self.include_intercept_ = include_intercept
        self.solver_ = solver
        self.lam_ = lam
        self.penalty_ = penalty
        self.alpha_ = alpha

        if penalty not in ["none", "l1", "l2"]:
            raise ValueError("Supported penalty types are: none, l1, l2")

        self.coefs_ = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Logistic regression model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model using specified `self.solver_` passed when instantiating class and includes an intercept
        if specified by `self.include_intercept_
        """
        # Add intercept term if specified
        if self.include_intercept_:
            X = np.c_[np.ones(len(X)), X]
            # intercept = np.ones((X.shape[0], 1)) #TODO אפשר להחליף את שתי שורות אלה בשורה שמעל
            # X = np.hstack((intercept, X))

        # Initialize weights
        init_weights = np.random.randn(X.shape[1]) / np.sqrt(X.shape[1])

        # Determine the objective based on the penalty type
        if self.penalty_ == "none":
            objective = LogisticModule(weights=init_weights)
        else:  # L1 or L2
            fidelity_module = LogisticModule()
            regularization_module = {"l1": L1, "l2": L2}[self.penalty_]()
            objective = RegularizedModule(fidelity_module=fidelity_module,
                                          regularization_module=regularization_module,
                                          lam=self.lam_,
                                          include_intercept=self.include_intercept_,
                                          weights=init_weights)

        # Fit the model using the solver
        self.coefs_ = self.solver_.fit(objective, X, y)

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        return self.predict_proba(X) >= self.alpha_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities of samples being classified as `1` according to sigmoid(Xw)

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict probability for

        Returns
        -------
        probabilities: ndarray of shape (n_samples,)
            Probability of each sample being classified as `1` according to the fitted model
        """
        # Ensure the estimator is fitted before making predictions
        if self.fitted_:
            # Add intercept term if specified
            if self.include_intercept_:
                X = np.c_[np.ones(len(X)), X]

            # Compute linear combination of input features and weights
            z = X @ self.coefs_

            # Compute the sigmoid of the linear combination
            sigmoid = 1 / (np.exp(-z) + 1)
            return sigmoid
        else:
            raise ValueError("The model must be fitted before calling predict_proba.")

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification error

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under misclassification error
        """
        # Predict responses
        y_pred = self._predict(X)

        # Calculate misclassification error using the provided function
        return misclassification_error(y_true=y, y_pred=y_pred)
