import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import machine_learning as ml


def visualization_data(dataset, x_label, y_label, x_title, y_title, title):
    x = dataset[x_label].values
    y = dataset[y_label].values
    b, a = ml.linear_regression(x, y)
    y_pred = b + a * x

    fig, ax = plt.subplots(1, figsize=(12, 10))
    sns.set(rc={'axes.facecolor': '#33FFA2',
                'figure.facecolor': 'lightgrey'})
    sns.lineplot(dataset.set_index(x_label)[y_label],
                 ax=ax)
    ax.plot(x, y_pred, color="r")
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    ax.set_title(title)

    return fig, ax