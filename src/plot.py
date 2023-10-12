import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np

DIV_LINE_WIDTH = 50


# def plot_data(
#     data,
#     xaxis="Epoch",
#     value="AverageEpRet",
#     condition="Condition1",
#     smooth=1,
#     **kwargs
# ):
#     if smooth > 1:
#         """
#         smooth data with moving window average.
#         that is,
#             smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
#         where the "smooth" param is width of that window (2k+1)
#         """
#         y = np.ones(smooth)
#         for datum in data:
#             x = np.asarray(datum[value])
#             z = np.ones(len(x))
#             smoothed_x = np.convolve(x, y, "same") / np.convolve(z, y, "same")
#             datum[value] = smoothed_x

#     if isinstance(data, list):
#         data = pd.concat(data, ignore_index=True)
#     sns.set(style="darkgrid", font_scale=1.5)
#     sns.lineplot(data=data, x=xaxis, y=value, hue=condition, ci="sd", **kwargs)
#     plt.legend(loc="best").set_draggable(True)
#     # plt.legend(loc='upper center', ncol=3, handlelength=1,
#     #           borderaxespad=0., prop={'size': 13})

#     # """
#     # For the version of the legend used in the Spinning Up benchmarking page,
#     # swap L38 with:

#     # plt.legend(loc='upper center', ncol=6, handlelength=1,
#     #            mode="expand", borderaxespad=0., prop={'size': 13})
#     # """

#     xscale = np.max(np.asarray(data[xaxis])) > 5e3
#     if xscale:
#         # Just some formatting niceness: x-axis scale in scientific notation if max x is large
#         plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

#     plt.tight_layout(pad=0.5)


# def make_plots(
#     all_logdirs,
#     legend=None,
#     xaxis=None,
#     values=None,
#     count=False,
#     font_scale=1.5,
#     smooth=1,
#     select=None,
#     exclude=None,
#     estimator="mean",
# ):
#     # data = get_all_datasets(all_logdirs, legend, select, exclude)
#     values = values if isinstance(values, list) else [values]
#     condition = "Condition2" if count else "Condition1"
#     estimator = getattr(
#         np, estimator
#     )  # choose what to show on main curve: mean? max? min?
#     for value in values:
#         plt.figure()
#         plot_data(
#             data,
#             xaxis=xaxis,
#             value=value,
#             condition=condition,
#             smooth=smooth,
#             estimator=estimator,
#         )
#     plt.show()


# def main():
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument("logdir", nargs="*", default="./runs")
#     parser.add_argument("--legend", "-l", nargs="*")
#     parser.add_argument("--xaxis", "-x", default="TotalEnvInteracts")
#     parser.add_argument("--value", "-y", default="Performance", nargs="*")
#     parser.add_argument("--count", action="store_true")
#     parser.add_argument("--smooth", "-s", type=int, default=1)
#     parser.add_argument("--select", nargs="*")
#     parser.add_argument("--exclude", nargs="*")
#     parser.add_argument("--est", default="mean")
#     args = parser.parse_args()
#     """

#     Args:
#         logdir (strings): As many log directories (or prefixes to log
#             directories, which the plotter will autocomplete internally) as
#             you'd like to plot from.

#         legend (strings): Optional way to specify legend for the plot. The
#             plotter legend will automatically use the ``exp_name`` from the
#             config.json file, unless you tell it otherwise through this flag.
#             This only works if you provide a name for each directory that
#             will get plotted. (Note: this may not be the same as the number
#             of logdir args you provide! Recall that the plotter looks for
#             autocompletes of the logdir args: there may be more than one
#             match for a given logdir prefix, and you will need to provide a
#             legend string for each one of those matches---unless you have
#             removed some of them as candidates via selection or exclusion
#             rules (below).)

#         xaxis (string): Pick what column from data is used for the x-axis.
#              Defaults to ``TotalEnvInteracts``.

#         value (strings): Pick what columns from data to graph on the y-axis.
#             Submitting multiple values will produce multiple graphs. Defaults
#             to ``Performance``, which is not an actual output of any algorithm.
#             Instead, ``Performance`` refers to either ``AverageEpRet``, the
#             correct performance measure for the on-policy algorithms, or
#             ``AverageTestEpRet``, the correct performance measure for the
#             off-policy algorithms. The plotter will automatically figure out
#             which of ``AverageEpRet`` or ``AverageTestEpRet`` to report for
#             each separate logdir.

#         count: Optional flag. By default, the plotter shows y-values which
#             are averaged across all results that share an ``exp_name``,
#             which is typically a set of identical experiments that only vary
#             in random seed. But if you'd like to see all of those curves
#             separately, use the ``--count`` flag.

#         smooth (int): Smooth data by averaging it over a fixed window. This
#             parameter says how wide the averaging window will be.

#         select (strings): Optional selection rule: the plotter will only show
#             curves from logdirs that contain all of these substrings.

#         exclude (strings): Optional exclusion rule: plotter will only show
#             curves from logdirs that do not contain these substrings.

#     """

#     make_plots(
#         args.logdir,
#         args.legend,
#         args.xaxis,
#         args.value,
#         args.count,
#         smooth=args.smooth,
#         select=args.select,
#         exclude=args.exclude,
#         estimator=args.est,
#     )


from tensorboard.backend.event_processing import event_accumulator
import pandas as pd


def parse_tensorboard(path, scalars):
    """returns a dictionary of pandas dataframes for each requested scalar"""
    ea = event_accumulator.EventAccumulator(
        path,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # make sure the scalars are in the event accumulator tags
    assert all(
        s in ea.Tags()["scalars"] for s in scalars
    ), "some scalars were not found in the event accumulator"
    df = {
        k: pd.DataFrame(ea.Scalars(k)).drop(columns=["wall_time"]).iloc[0:72]
        for k in scalars
    }
    return df


def plot_data(
    data,
    xaxis,
    yaxis,
    smooth=10,
):
    if smooth > 1:
        # """
        # smooth data with moving window average.
        # that is,
        #     smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        # where the "smooth" param is width of that window (2k+1)
        # """
        y = np.ones(smooth)
        x = np.asarray(data[yaxis])
        z = np.ones(len(x))
        smoothed_x = np.convolve(x, y, "same") / np.convolve(z, y, "same")
        data[yaxis] = smoothed_x
    # plt.figure()
    sns.set(style="darkgrid", font_scale=1.5)
    plt.legend(loc="best").set_draggable(True)
    sns.lineplot(data=data, x=xaxis, y=yaxis, errorbar="sd")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    plt.tight_layout(pad=0.5)


def make_plots(data_list, xaxis, yaxis, figname):
    plt.figure(figsize=(10, 5))
    for data in data_list:
        plot_data(data, xaxis, yaxis)
    plt.savefig(figname, dpi=400)


def get_all_data(logdir, scalars):
    logdir = osp.abspath(logdir)
    listdir = os.listdir(logdir)
    listdir = [osp.join(logdir, listdir[i]) for i in range(len(listdir))]
    df_dict = {}
    for exp in listdir:
        suffix = exp.split("/")[-1]
        df_dict[suffix] = parse_tensorboard(exp, scalars)
        # df_dict[suffix] = df_dict[suffix][scalars[0]].iloc[:72]
    return df_dict


def data_agg(df_dict, scalars, figname):
    for scalar in scalars:
        data_list = []
        for exp in df_dict.keys():
            data_list.append(df_dict[exp][scalar])
        make_plots(data_list, "step", "value", figname)


# scalars = ["eval/episodic_r", "eval/episodic_r_carb", "eval/episodic_r_elec"]


# data_agg(df_dict, scalars, figname="episodic_r.pdf")


# data_agg(df_dict, scalars, figname="episodic_r_carb.pdf")


# data_agg(df_dict, scalars, figname="episodic_r_elec.pdf")
scalars = ["eval/episodic_r"]
df_dict = get_all_data("runs", scalars)

df_tmp = pd.DataFrame()
for exp in df_dict.keys():
    df_dict[exp][scalars[0]]["exp"] = exp
    df_tmp = (
        pd.concat([df_tmp, df_dict[exp][scalars[0]]], axis=0)
        .sort_values(by="step")
        .reset_index(drop=True)
    )

df_tmp_rule = pd.DataFrame()
df_tmp_rule["step"] = df_tmp["step"].unique()
df_tmp_rule["value"] = np.ones(len(df_tmp["step"].unique())) * 170.14470398426056
df_tmp_rule["exp"] = "rule-based"

plt.figure(figsize=(10, 4))
data = df_tmp
smooth = 3
xaxis = "step"
yaxis = "value"
figname = "episodic_r.pdf"
if smooth > 1:
    y = np.ones(smooth)
    x = np.asarray(data[yaxis])
    z = np.ones(len(x))
    smoothed_x = np.convolve(x, y, "same") / np.convolve(z, y, "same")
    data[yaxis] = smoothed_x
# plt.figure()
sns.set(style="darkgrid", font_scale=1.5)
plt.legend(loc="best").set_draggable(True)
sns.lineplot(data=data, x=xaxis, y=yaxis, errorbar="sd", label="MM-PPO")
sns.lineplot(data=df_tmp_rule, x=xaxis, y=yaxis, label="Competitive + Rule-Based")
plt.xlabel("Learning step")
plt.ylabel("Multi-market reward")
xscale = np.max(np.asarray(data[xaxis])) > 5e3
if xscale:
    # Just some formatting niceness: x-axis scale in scientific notation if max x is large
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
# Add the annotation
plt.annotate(
    "Episodic cumulative reward is 170.144",
    xy=(10000, 170.144),
    xytext=(14000 - 0.1, 170.144 + 5000),
    arrowprops=dict(facecolor="red", edgecolor="red", arrowstyle="->", linewidth=2),
    horizontalalignment="right",
)
plt.tight_layout(pad=0.5)
plt.savefig(figname, dpi=400)


# carb reward
scalars = ["eval/episodic_r_carb"]
df_dict = get_all_data("runs", scalars)

df_tmp = pd.DataFrame()
for exp in df_dict.keys():
    df_dict[exp][scalars[0]]["exp"] = exp
    df_tmp = (
        pd.concat([df_tmp, df_dict[exp][scalars[0]]], axis=0)
        .sort_values(by="step")
        .reset_index(drop=True)
    )

reward = -159.728

df_tmp_rule = pd.DataFrame()
df_tmp_rule["step"] = df_tmp["step"].unique()
df_tmp_rule["value"] = np.ones(len(df_tmp["step"].unique())) * reward
df_tmp_rule["exp"] = "rule-based"

plt.figure(figsize=(10, 4))
data = df_tmp
smooth = 3
xaxis = "step"
yaxis = "value"
figname = "episodic_r_carb.pdf"
if smooth > 1:
    y = np.ones(smooth)
    x = np.asarray(data[yaxis])
    z = np.ones(len(x))
    smoothed_x = np.convolve(x, y, "same") / np.convolve(z, y, "same")
    data[yaxis] = smoothed_x
# plt.figure()
sns.set(style="darkgrid", font_scale=1.5)
plt.legend(loc="best").set_draggable(True)
sns.lineplot(data=data, x=xaxis, y=yaxis, errorbar="sd", label="MM-PPO")
sns.lineplot(data=df_tmp_rule, x=xaxis, y=yaxis, label="Rule-based")
plt.xlabel("Learning step")
plt.ylabel("Carbon market reward")
xscale = np.max(np.asarray(data[xaxis])) > 5e3
if xscale:
    # Just some formatting niceness: x-axis scale in scientific notation if max x is large
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
# Add the annotation
plt.annotate(
    f"Episodic cumulative reward is {reward:.3f}",
    xy=(10000, reward),
    xytext=(14000 - 0.1, reward + 5000),
    arrowprops=dict(facecolor="red", edgecolor="red", arrowstyle="->", linewidth=2),
    horizontalalignment="right",
)
plt.tight_layout(pad=0.5)
plt.savefig(figname, dpi=400)


# elec

scalars = ["eval/episodic_r_elec"]
df_dict = get_all_data("runs", scalars)
df_tmp = pd.DataFrame()
for exp in df_dict.keys():
    df_dict[exp][scalars[0]]["exp"] = exp
    df_tmp = (
        pd.concat([df_tmp, df_dict[exp][scalars[0]]], axis=0)
        .sort_values(by="step")
        .reset_index(drop=True)
    )

reward = 329.8736621141434

df_tmp_rule = pd.DataFrame()
df_tmp_rule["step"] = df_tmp["step"].unique()
df_tmp_rule["value"] = np.ones(len(df_tmp["step"].unique())) * reward
df_tmp_rule["exp"] = "Competitive"

plt.figure(figsize=(10, 4))
data = df_tmp
smooth = 3
xaxis = "step"
yaxis = "value"
figname = "episodic_r_elec.pdf"
if smooth > 1:
    y = np.ones(smooth)
    x = np.asarray(data[yaxis])
    z = np.ones(len(x))
    smoothed_x = np.convolve(x, y, "same") / np.convolve(z, y, "same")
    data[yaxis] = smoothed_x
# plt.figure()
sns.set(style="darkgrid", font_scale=1.5)
plt.legend(loc="best").set_draggable(True)
sns.lineplot(data=data, x=xaxis, y=yaxis, errorbar="sd", label="MM-PPO")
sns.lineplot(data=df_tmp_rule, x=xaxis, y=yaxis, label="Competitive")
plt.xlabel("Learning step")
plt.ylabel("Electricity market eeward")
xscale = np.max(np.asarray(data[xaxis])) > 5e3
if xscale:
    # Just some formatting niceness: x-axis scale in scientific notation if max x is large
    plt.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
# Add the annotation
plt.annotate(
    f"Episodic cumulative reward is {reward:.3f}",
    xy=(10000, reward),
    xytext=(14000 - 0.1, reward - 50),
    arrowprops=dict(facecolor="red", edgecolor="red", arrowstyle="->", linewidth=2),
    horizontalalignment="right",
)
plt.tight_layout(pad=0.5)
plt.savefig(figname, dpi=400)
