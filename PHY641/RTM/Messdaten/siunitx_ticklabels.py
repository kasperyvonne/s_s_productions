import matplotlib.pyplot as plt
import numpy as np

if plt.rcParams["text.usetex"] is False:
    plt.rcParams["text.usetex"] = True

if plt.rcParams["text.latex.unicode"] is False:
    plt.rcParams["text.latex.unicode"] = True

if "siunitx" not in plt.rcParams["text.latex.preamble"]:
    plt.rcParams["text.latex.preamble"].append(r"\usepackage{siunitx}")



def siunitx_ticklabels(ax=None, locale="DE", xaxis=True, yaxis=True):

    if ax is None:
        ax = plt.gca()

    if xaxis is True:
        xticks = ax.get_xticks()
        xlabels = [r"$\num[locale={}]{{{}}}$".format(locale, tick) for tick in xticks]
        ax.set_xticklabels(xlabels)

    if yaxis is True:
        yticks = ax.get_yticks()
        ylabels = [r"$\num[locale={}]{{{}}}$".format(locale, tick) for tick in yticks]
        ax.set_yticklabels(ylabels)
