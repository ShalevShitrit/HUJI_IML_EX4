import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from base_module import BaseModule
from base_learning_rate import BaseLR
from gradient_descent import GradientDescent
from learning_rate import FixedLR

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
    v = []
    w = []

    def callback(val, weight, **kwargs):
        v.append(val)
        w.append(weight)

    return (callback, v, w)


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
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
        vals = []
        weights = []

        def callback(val, weight, **kwargs):
            vals.append(val)
            weights.append(weight)

        return (callback, vals, weights)

    def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                     etas: Tuple[float] = (1, .1, .01, .001)):
        for i in {1, 2}:
            # Select module (L1 and then L2)
            module = L1 if i == 1 else L2
            # Dictionary to store recorded values and weights per eta
            results = dict()

            for eta in etas:
                # Retrieve a new callback function
                callback, vals, weights = get_gd_state_recorder_callback()

                # Create a GradientDescent instance with the specified learning rate
                GD = GradientDescent(learning_rate=FixedLR(eta), callback=callback)
                # Fit the model to minimize the given objective with the initial weights
                GD.fit(module(weights=np.copy(init)), None, None)
                # Store the recorded values and weights for the current learning rate
                results[eta] = (vals, weights)

                # Plot algorithm's descent path
                title = f"L{i}: Learning-Rate={eta}"
                file_path = f"../plots/GD_L{i}_eta_{eta}.png"
                plot_descent_path(module, np.array([init] + weights), title).write_image(file_path)

            # Plot algorithm's convergence for the different values of eta
            x_title = "Gradient-Descent iteration"
            y_title = "Norm"
            title = f"L{i} Gradient-Descent convergence for different Learning-Rates"
            layout = go.Layout(xaxis=dict(title=x_title), yaxis=dict(title=y_title), title=title)
            fig = go.Figure(layout)
            mode = "lines"
            name = rf"$\eta={eta}$"
            # Add traces for each learning rate's convergence
            for eta, (v, _) in results.items():
                fig.add_trace(go.Scatter(x=list(range(len(v))), y=v, mode=mode, name=name))

            # Save the convergence plot image
            file_path = f"../plots/GD_L{i}_fixed_rate_convergence.png"
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
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    raise NotImplementedError()

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    fit_logistic_regression()
