import numpy as np
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


def chart_pie(labels, sizes):
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    explode = (0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
           shadow=True, startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    return fig, ax
