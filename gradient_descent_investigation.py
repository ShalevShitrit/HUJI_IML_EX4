import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

import sklearn
from sklearn.metrics import auc

import cross_validate
from base_module import BaseModule
from base_learning_rate import BaseLR
from gradient_descent import GradientDescent
from learning_rate import FixedLR
from loss_functions import misclassification_error

# from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from modules import L1, L2
from logistic_regression import LogisticRegression
from utils import split_train_test

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure(
        [decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
         go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                    marker_color="black")],
        layout=go.Layout(xaxis=dict(range=xrange),
                         yaxis=dict(range=yrange),
                         title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[
    Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    # Lists to store the recorded values and weights
    values = []
    weights_ = []

    # Define the callback function
    def callback(weights: np.ndarray, val: np.ndarray, **kwargs) -> None:
        weights_.append(weights)  # Append the current weights to the list
        values.append(val)  # Append the current value of the objective function to the list

    # Return the callback function and the lists
    return callback, values, weights_


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    print('-------------------------------------------------------------------------')
    print('                       Compare Fixed Learning Rates                      ')
    print('-------------------------------------------------------------------------')

    print('\n> Creating GD plots for l1 and l2:')
    for i in {1, 2}:
        # Select module (L1 and then L2)
        model = L1 if i == 1 else L2
        # Dictionary to store recorded values and weights per eta
        results = dict()

        for eta in etas:
            print(f"  > Creating plot with L{i} and eta={eta}.")
            # Retrieve a new callback function
            callback, vals, weights = get_gd_state_recorder_callback()

            # Create a GradientDescent instance with the specified learning rate
            GD = GradientDescent(learning_rate=FixedLR(eta), callback=callback)
            # Fit the model to minimize the given objective with the initial weights
            GD.fit(model(np.copy(init)), None, None)
            # Store the recorded values and weights for the current learning rate
            results[eta] = (vals, weights)

            # Plot algorithm's descent path
            title = f"L{i} (Learning-Rate={eta})"
            fig = plot_descent_path(model, np.array([init] + weights), title)
            file_path = f"plots/GD_L{i}_eta_{eta}.png"
            fig.write_image(file_path)

        print(f"    > Creating plot for convergent rate with L{i}.\n")
        # Plot algorithm's convergence for the different values of eta
        x_title = "Gradient-Descent iteration"
        y_title = "Norm"
        title = f"L{i} Gradient-Descent convergence for different Learning-Rates"
        layout = go.Layout(xaxis=dict(title=x_title), yaxis=dict(title=y_title), title=title)
        fig = go.Figure(layout=layout)
        mode = "lines"

        # Add traces for each learning rate's convergence
        for eta, (v, _) in results.items():
            name = rf"$\eta={eta}$"
            x_scatter = list(range(len(v)))
            go_scatter = go.Scatter(x=x_scatter, y=v, mode=mode, name=name)
            fig.add_trace(go_scatter)

        # Generate the convergence plot image
        file_path = f"plots/GD_L{i}_fixed_rate_convergence.png"
        fig.write_image(file_path)


def load_data(path: str = "SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    print('-------------------------------------------------------------------------')
    print('                         Fit Logistic Regression                         ')
    print('-------------------------------------------------------------------------')

    # Load and split SA Heart Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # PART 1 - Plotting convergence rate of logistic regression over SA heart disease data

    # Get a callback function for recording gradient descent state
    callback, values, weights = get_gd_state_recorder_callback()

    # Instantiate the gradient descent solver with fixed learning rate and maximum iterations
    fixed_lr = FixedLR(1e-4)
    max_iter = 20000
    solver = GradientDescent(learning_rate=fixed_lr, max_iter=max_iter, callback=callback)

    # Instantiate and fit the logistic regression model using the gradient descent solver
    model = LogisticRegression(solver=solver)
    model.fit(X_train.values, y_train.values)

    # Predict probabilities on the training set
    y_proba_prediction = model.predict_proba(X_test.values)

    # Compute ROC curve and ROC area using true labels and predicted probabilities
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_proba_prediction)

    # Plot ROC curve
    print('\n> Creating ROC curve plot.')
    roc_auc = auc(fpr, tpr)
    plot_ROC_curve(fpr, roc_auc, tpr)

    # Calculate the optimal alpha threshold
    print('\n> Calculating optimal alpha threshold:')
    calculate_optimal_threshold(X_test, fpr, model, thresholds, tpr, y_test)

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify
    # values of regularization parameter

    # Define a list of lambda values for cross-validation
    print('\n> Checking Linear-Regression with different lambdas values:')
    lambdas = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]

    # Initialize an array to store cross-validation results
    res = np.zeros((len(lambdas), 2))

    # Perform cross-validation for each lambda value
    for i, lam in enumerate(lambdas):
        solver = GradientDescent(max_iter=max_iter, learning_rate=fixed_lr)
        LG = LogisticRegression(alpha=0.5, solver=solver, penalty="l1", lam=lam)
        res[i] = cross_validate.cross_validate(estimator=LG, X=X_train.values, y=y_train.values,
                                               scoring=misclassification_error)
        lam = "{:.3f}".format(lam)
        train_e = "{:.3f}".format(res[i, 0])
        test_e = "{:.3f}".format(res[i, 1])
        print(f'  > penalty: L1 | lambda: {lam} | train error: {train_e} | test error: {test_e}.')

    # Plot training and validation errors for different lambda values
    title = r"$\text{Train and Validation errors (averaged over the k-folds)}$"
    xaxis = dict(title=r"$\lambda$", type="log")
    yaxis_title = r"$\text{Error Value}$"
    fig_layout = go.Layout(title=title, xaxis=xaxis, yaxis_title=yaxis_title)
    fig = go.Figure([go.Scatter(x=lambdas, y=res[:, 0], name="Train-Error"),
                     go.Scatter(x=lambdas, y=res[:, 1], name="Validation-Error")],
                    layout=fig_layout)

    # Save cross-validation plot to a PNG file
    file_path = f"plots/LR_cross_validation.png"
    fig.write_image(file_path)

    # Find the optimal lambda value that minimizes validation error
    lam_opt = lambdas[np.argmin(res[:, 1])]

    # Instantiate the gradient descent solver for the final model
    GD = GradientDescent(learning_rate=fixed_lr, max_iter=max_iter)

    # Fit logistic regression with optimal lambda on the entire training set
    model = (LogisticRegression(solver=GD, penalty="l1", lam=lam_opt, alpha=.5).
             fit(X_train.values, y_train.values))

    # Print the optimal regularization parameter and the test error of the final model
    print(f'\n> Optimal regularization parameter: {lam_opt}.')
    print(f'  > Model test error with this parameter: {round(model.loss(X_test.values, y_test.values), 2)}.')


def plot_ROC_curve(fpr, roc_auc, tpr):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (area = {roc_auc:.2f})'))

    # Add a dashed diagonal line to represent a random classifier
    line_dict = dict(dash='dash')
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=line_dict))

    # Update plot layout with titles and axis labels
    title_layout = 'Receiver Operating Characteristic (ROC)'
    xaxis_layout = 'False-Positive Rate'
    yaxis_layout = 'True-Positive Rate'
    fig.update_layout(title=title_layout, xaxis_title=xaxis_layout, yaxis_title=yaxis_layout)

    # Save ROC curve plot to a PNG file
    file_path = f"plots/LR_ROC_curve.png"
    fig.write_image(file_path)


def calculate_optimal_threshold(X_test, fpr, model, thresholds, tpr, y_test):
    # Find the index of the optimal threshold where the difference between TPR and FPR is maximized
    optimal_idx = np.argmax(tpr - fpr)

    # Get the optimal threshold value
    optimal_alpha = thresholds[optimal_idx]
    print(f'  > Optimal TPR-FPR threshold: {optimal_alpha}.')

    # Update the model with the optimal alpha
    model.alpha_ = optimal_alpha

    # Compute and print the test error at the optimal alpha
    test_error = model._loss(X_test.values, y_test.values)
    print(f'  > Test error with this threshold: {test_error}.')


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    fit_logistic_regression()
