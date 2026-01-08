from typing import Literal, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass


def autogrid(
    ax: plt.Axes,
    axises: Union[Literal["x"], Literal["y"], Literal["all"]] = "y",
    which: Union[Literal["major"], Literal["minor"], Literal["all"], Literal["none"]] = "all",
    minor_tick: bool = True,
    step=5,
):
    def _auto(ax, axis):
        axis = getattr(ax, f"{axis}axis")
        major_locator = axis.get_major_locator()
        major_step = major_locator()[1] - major_locator()[0]
        if which in ["major", "all"]:
            axis.grid(True, "major", color="#ccc", linewidth=0.8, zorder=-100)
        if which in ["minor", "all"] or minor_tick:
            minor_step = major_step / step
            axis.set_minor_locator(plt.MultipleLocator(minor_step))
        if which in ["minor", "all"]:
            axis.grid(True, "minor", color="#eee", linewidth=0.8, zorder=-100, linestyle="--")

    if axises == "all":
        for axis in ["x", "y"]:
            _auto(ax, axis)
    else:
        _auto(ax, axises)


if __name__ == "__main__":

    plt.rcParams.update({
        "lines.linewidth": 1.5,
        # "lines.markersize": 5,
        "axes.labelweight": "bold",
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "lines.markerfacecolor": "auto",
        "lines.markeredgecolor": "auto",
        "font.size": 10,
        "font.family": "sans-serif",
    })

    df = pd.read_json("results/result.json", lines=True)
    df["throughput"] = df["completed"] / df["duration"]
    for file in range(5):
        if file < 4:
            fig, axes = plt.subplots(1, 3, figsize=(15, 3), sharey=False, gridspec_kw={"wspace": 0.3})
        else:
            fig, axes = plt.subplots(1, 1, figsize=(5, 3), sharey=False, gridspec_kw={"wspace": 0.3})
        axes = np.atleast_1d(axes)
        name = ["e2e_latency", "ttft", "tpot", "itl", "throughput"][file]
        for col in range(len(axes)):
            ax = axes[col]
            metric = ["mean", "median", "p99"][col]
            yname = f"{metric}_{name}_ms" if name != "throughput" else name
            sns.lineplot(df, x="request_rate", y=yname, hue="backend", ax=ax, marker="X")
            ax.set_xlabel("Request Rate")
            ax.set_title(metric.capitalize())
            autogrid(ax)
            if col == 0:
                ax.set_ylabel(" ".join(yname.split("_")).capitalize())
            else:
                ax.set_ylabel("")
            if file == 4:
                ax.set_title("Throughput")
                ax.set_ylabel("req/s")
        fig.savefig(f"results/{name}.pdf", bbox_inches='tight')
        print(f"save figure at results/{name}.pdf")
