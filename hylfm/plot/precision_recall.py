import typing
from argparse import ArgumentParser
from collections import OrderedDict, defaultdict
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn
from ruamel.yaml import YAML

from hylfm.plot.metrics import get_datasets

yaml = YAML(typ="safe")


def add_plot(plot_name, axes, out_dir):
    plt.rcParams['svg.fonttype'] = 'none'
    seaborn.set_context("talk")

    metric_postfix = ""
    use_local_data = False

    if plot_name == "heart_static_1":
        along_z = "-along_z"
        scalar_data = [
            ["hylfm_dyn", "scaled-True"],
            ["hylfm_stat", "scaled-True"],
            ["lr", "scaled-True"],
            # ["v0_on_48x88x88", "scaled-True"],
            # ["hylfm_dyn", "scaled-False"],
            # ["hylfm_stat", "scaled-False"],
            # ["v0_on_48x88x88", "scaled-False"],
        ]
        z_slice_mod: typing.Optional[int] = None  # 209 None
        nets = ["lr", "hylfm_dyn", "hylfm_stat", "v0_on_48x88x88"]
        if use_local_data:
            test_data_path = Path("C:/Users/fbeut/Desktop/lnet_stuff/plots/data/heart/static1")
        else:
            test_data_path = Path("K:/LF_computed/lnet/plain/heart/static1/test")
    elif plot_name == "heart_dynamic_1" and not use_local_data:
        along_z = ""
        scalar_data = [
            ["hylfm_dyn", "scaled-True"],
            ["hylfm_stat", "scaled-True"],
            ["lr", "scaled-True"],
            # ["v0_on_48x88x88", "scaled-True"],
            # ["hylfm_dyn", "scaled-False"],
            # ["hylfm_stat", "scaled-False"],
            # ["v0_on_48x88x88", "scaled-False"],
        ]
        z_slice_mod: typing.Optional[int] = 209  # 209 None
        nets = ["lr", "hylfm_dyn", "hylfm_stat", "v0_on_48x88x88"]
        test_data_path = Path("K:/LF_computed/lnet/plain/heart/dynamic1/test/metric_copy")
    elif plot_name.startswith("beads") and not use_local_data:
        along_z = "_along_z"
        if plot_name == "beads":
            z_slice_mod: typing.Optional[int] = None  # 209 None
            scalar_data = [["hylfm", "scaled-True"], ["lr", "scaled-True"]]
            scalar_data.append(["v0_on_56x80x80", "scaled-True"])
        elif plot_name == "beads_pr":
            z_slice_mod: typing.Optional[int] = 51  # 209 None
            scalar_data = []
            tgt_threshold = "0.1"
            # for threshold in ["1.0", "0.75", "0.5", "0.25", "0.1", "0.075", "0.05", "0.025", "0.01", "0.0075", "0.005", "0.0025", "0.001", "0.0005", "0.0001"]:
            for threshold in [
                "0.75",
                "0.5",
                "0.25",
                "0.1",  # a
                "0.075",
                "0.05",  # b
                "0.025",
                "0.01",  # c
                "0.0075",
                "0.005",  # d
                "0.0025",
                "0.001",
                "0.0005",
                "0.0001",
            ]:
                hyper_parameter_name = f"distthreshold-3.0_excludeborder-False_maxsigma-6.0_minsigma-1.0_overlap-0.5_sigmaratio-3.0_threshold-{threshold}_tgtthreshold-{tgt_threshold}_scaled-True"
                scalar_data.append(["hylfm", hyper_parameter_name])
                scalar_data.append(["lr", hyper_parameter_name])
                scalar_data.append(["v0_on_56x80x80", hyper_parameter_name])
        else:
            raise NotImplementedError(plot_name)

        nets = ["lr", "hylfm", "v0_on_56x80x80"]
        test_data_path = Path("K:/LF_computed/lnet/plain/beads/f8_01highc/test")
    else:
        raise NotImplementedError(plot_name)

    plot_name += metric_postfix + ""
    title = ""  #  f"{net} {metric} on static heart"
    map_labels = {}
    all_paths = {}
    for net in nets:
        all_paths[net] = {}
        if use_local_data:
            result_path = test_data_path / net
        else:
            result_path = test_data_path / net / "metrics"

        all_paths[net] = {}
        for metric_name in ["PSNR", "NRMSE", "SSIM", "MS-SSIM", "MSE", "Smooth L1", "Precision", "Recall"]:
            all_paths[net][metric_name] = {}
            if metric_name in ["MSE", "Smooth L1"]:
                in_file_metric_name = metric_name + "_loss"
            else:
                in_file_metric_name = metric_name

            in_file_metric_name = in_file_metric_name.replace("-", "_").replace(" ", "_")
            for scaled in [False, True]:
                ifmn = in_file_metric_name + metric_postfix
                if scaled:
                    ifmn += "-scaled"

                ifmn += along_z
                file_name = f"{ifmn.lower()}.yml"
                all_paths[net][metric_name][f"scaled-{scaled}"] = result_path / file_name

    for net in nets:
        if use_local_data:
            result_path = test_data_path / net
        else:
            result_path = test_data_path / net / "metrics"

        for metric_name in ["Precision", "Recall"]:
            all_paths[net][metric_name] = {}
            metric_file_name_start = f"bead_{metric_name.lower()}{along_z}-"
            # print('search in ', result_path / metric_file_name_start)
            for file_path in result_path.glob(f"{metric_file_name_start}*.yml"):
                hyperparameter_name = file_path.stem[len(metric_file_name_start) :]
                if hyperparameter_name.endswith("-scaled"):
                    hyperparameter_name = hyperparameter_name.replace("-scaled", "_scaled-True")
                else:
                    hyperparameter_name += "_scaled-False"

                # print('adding', hyperparameter_name)
                all_paths[net][metric_name][hyperparameter_name] = file_path

    # pprint(all_paths)

    better_network_names = {
        "hylfm": "HyLFM-Net",
        "hylfm_dyn": "HyLFM-Net dyn",
        "hylfm_stat": "HyLFM-Net stat",
        "v0_on_48x88x88": "LFD+CARE",
        "v0_on_56x80x80": "LFD+CARE",
        "lr": "LFD",
    }
    # palette = {
    #     "HyLFM-Net": "#648FFF",
    #     "HyLFM-Net stat": "#648FFF",
    #     "HyLFM-Net dyn": "#785EF0",
    #     "LFD+CARE": "#DC267F",
    #     "LFD": "#FFB000",
    # }  # blue: #648FFF, purple: #785EF0, red: #DC267F, dark orange: #FE6100, orange: #FFB000
    palette = {
        "HyLFM-Net": "#DC267F",
        "HyLFM-Net stat": "#648FFF",
        "HyLFM-Net dyn": "#785EF0",
        "LFD+CARE": "#785EF0",
        "LFD": "#FFB000",
    }  # blue: #648FFF, purple: #785EF0, red: #DC267F, dark orange: #FE6100, orange: #FFB000

    hue_order = None
    # [
    #     "LFD+CARE",
    #     "LFD",
    #     "HyLFM-Net",
    #     "HyLFM-Net dyn",
    #     "HyLFM-Net stat",
    # ]
    # legend_loc = {
    #     "PSNR": "upper right",
    #     "NRMSE": None, #"off",
    #     "SSIM": None, #"off",
    #     "MS-SSIM": None, #"off",
    #     "MSE": None, #"off",
    #     "Smooth L1": None, #"off",
    # }
    legend_loc = {
        "PSNR": "off"
        if plot_name.startswith("beads")
        else "upper center"
        if plot_name.startswith("heart_dynamic_1")
        else "upper right",
        "NRMSE": "off",
        "SSIM": "off",
        "MS-SSIM": "lower right" if plot_name.startswith("beads") else "off",
        "MSE": "off",
        "Smooth L1": "off",
        "Precision": None,
        "Recall": None,
    }

    # fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
    # fig, axes = plt.subplots(2, 3, figsize=(9, 5))

    scalar_columns = ["network", "hyperparameter_name", "key"]

    # plot_name += "overview"
    # metrics = ["PSNR", "MSE", "NRMSE", "SSIM", "MS-SSIM", None]
    # metrics = [
    #     "MS-SSIM",
    #     "SSIM",
    #     "PSNR",
    #     "NRMSE",
    #     # "MSE",
    #     # "Smooth L1",
    # ]  # , "PSNR", "MSE", "NRMSE", "SSIM", "MS-SSIM", None]
    # metrics = ["PSNR"]
    metrics = ["Precision", "Recall"]
    # legends = [None, {}, None, None, None, None]

    avg_values = {}
    # dfs = []
    df = pandas.DataFrame(columns=["key"])
    for i, metric_name in enumerate(metrics):
        scalar_data_here = [list(sd) for sd in scalar_data]
        scalar_columns_here = list(scalar_columns)

        for k, sd in enumerate(scalar_data_here):
            sd.append(k)  # key

        # ax = axes[i]
        if plot_name.startswith("heart_static_1"):
            if metric_name == "MSE":
                ax.set_ylim(0, 0.006)
            elif metric_name == "NRMSE":
                ax.set_ylim(0.2645, 1.6)
            elif metric_name == "PSNR":
                ax.set_ylim(19.8, 48)
            elif metric_name == "SSIM":
                ax.set_ylim(0.255, 1.0)
            elif metric_name == "MS-SSIM":
                ax.set_ylim(-0.1, 1.1)
            elif metric_name == "Smooth L1":
                ax.set_ylim(0, 0.006)
            else:
                raise NotImplementedError
        elif plot_name.startswith("heart_dynamic_1"):
            if metric_name == "MSE":
                ax.set_ylim(0, 0.006)
            elif metric_name == "NRMSE":
                ax.set_ylim(0.56, 1.01)
            elif metric_name == "PSNR":
                ax.set_ylim(23, 48)
            elif metric_name == "SSIM":
                ax.set_ylim(0.61, 1)
            elif metric_name == "MS-SSIM":
                ax.set_ylim(0.74, 1)
            elif metric_name == "Smooth L1":
                ax.set_ylim(0, 0.006)
            else:
                raise NotImplementedError

        if metric_name is None:
            ax.axis("off")
            continue

        # print(scalar_df)
        datasets, z_slice, z = get_datasets(all_paths, scalar_data_here, metric_name, z_slice_mod)
        # value_dfs = [pandas.DataFrame(numpy.asarray([numpy.repeat([k], len(dat)), dat, z_slice, z]).T, columns=["key", metric_name, "z slice", "z"]) for k, dat in enumerate(datasets)]
        value_dfs = [
            pandas.DataFrame(
                {
                    "key": numpy.repeat([k], len(dat)),
                    # "metric_name": metric_name,
                    metric_name: dat,
                    "z_slice": z_slice,
                    "z": z,
                }
            )
            for k, dat in enumerate(datasets)
        ]

        new_scalar_columns = [kv.split("-")[0] for kv in scalar_data_here[0][1].split("_")]
        for sd in scalar_data_here:
            sd[0] = better_network_names[sd[0]]
            hyperparameter_name = sd[1]
            for i, kv in enumerate(hyperparameter_name.split("_")):
                if kv.count("-") == 1:
                    k, v = kv.split("-")
                else:
                    raise NotImplementedError(hyperparameter_name, kv)

                assert new_scalar_columns[i] == k
                v = yaml.load(v)
                sd.append(v)

        scalar_columns_here += new_scalar_columns

        avg_values[metric_name] = {}
        for sd, vdf in zip(scalar_data_here, value_dfs):
            avg_values[metric_name][sd[0]] = vdf[metric_name].mean(), vdf[metric_name].std()

        print(metric_name)
        pprint(avg_values[metric_name])
        avg_values[metric_name] = None

        scalar_df = pandas.DataFrame(scalar_data_here, columns=scalar_columns_here)
        df = df.merge(scalar_df.merge(pandas.concat(value_dfs)), how="outer")

        # df = df.reset_index(drop=True)
        # print(df)
        # seaborn.set(style="ticks", color_codes=False)
        # seaborn.catplot(x="slice", y=first_name, kind="swarm", data=data)

        # size_order=["T1", "T2"]

        # plt.close(fg.fig)
        # g = seaborn.FacetGrid(df, col="threshold", height=1.5, hue="network", palette=palette)  # , col_wrap=2
        # g.map(plt.line,
        #     x=x_name,
        #     y=metric_name,
        #     data=df,
        #     # style="scaled",
        #     # style_order=[True, False],
        #     # legend="brief",
        #     # ci="sd",
        #     # ax=ax,
        # )  # , legend="full", ax=ax)

        # seaborn.lineplot(
        #     x=x_name,
        #     y=metric_name,
        #     data=df,
        #     hue="network",
        #     palette=palette,
        #     # style="scaled",
        #     # style_order=[True, False],
        #     legend="brief",
        #     ci="sd",
        #     ax=ax,
        # )  # , legend="full", ax=ax)

        # seaborn.despine(ax=ax)
        # if x_name == "z slice":
        #     ax.set_xlim(0, z_slice.max())
        # elif x_name == "z":
        #     ax.set(xlabel="z [μm]")
        #     ax.set_xlim(z.min(), z.max())
        #     # ax.set_xlim(-104, 104)
        #     # ax.set_xlim(-80, 80)
        #
        # # for name in metrics[slice(0, to_twin_at)]:
        # # print(df)
        #
        # # if to_twin_at is not None:
        # #     twinx = ax.twinx()
        # #     for name in metrics[to_twin_at:]:
        # #         seaborn.lineplot(x=x_name, y=name, data=df, legend="full", label=map_labels.get(name, name), ax=twinx)
        # #
        # #     twinx.set(ylabel=twin_y_label)
        #
        # ax.set_title(title)
        # handles, labels = ax.get_legend_handles_labels()
        # loc = legend_loc[metric_name]
        # if loc == "off":
        #     ax.get_legend().remove()
        # else:
        #     labels = labels[1:]  # remove 'network' category name
        #     if avg_values[metric_name] is not None:
        #         for i in range(len(labels)):
        #             mean, std = avg_values[metric_name][labels[i]]
        #             if metric_name in ["PSNR"]:
        #                 labels[i] += f" ({mean:.1f}±{std:.1f})"
        #             elif metric_name in ["MS-SSIM", "SSIM"]:
        #                 labels[i] += f" ({mean:.2}±{std:.2})"
        #             elif metric_name in ["NRMSE"]:
        #                 labels[i] += f" ({mean:.2f}±{std:.2f})"
        #             else:
        #                 raise NotImplementedError(metric_name)
        #
        #     ax.legend(handles=handles[1:], labels=labels, loc=loc)
        #
        # ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    # g = seaborn.FacetGrid(df, col="threshold", row="metric_name",
    #                    margin_titles=True)
    # g = g.map(plt.plot, x_name, "metric_value")#, label="metric_name")
    # # g = g.map_dataframe(seaborn.relplot, x_name, "metric_value", hue="network")#, label="metric_name")
    # # g = g.map_dataframe(seaborn.relplot, x_name, "metric_value", hue="network")#, label="metric_name")
    # # g.set(xlim=(0, 60), ylim=(0, 12),
    # #       xticks=[10, 30, 50], yticks=[2, 6, 10])
    # g.fig.subplots_adjust(wspace=.05, hspace=.05)
    #
    # # Iterate thorugh each axis
    # for ax in g.axes.flat:
    #     ax.set_title(ax.get_title(), fontsize='xx-large')
    #     # This only works for the left ylabels
    #     ax.set_ylabel(ax.get_ylabel(), fontsize='xx-large')

    # g = seaborn.relplot(
    #     x="Recall",
    #     y="Precision",
    #     hue="network",
    #     # size="threshold",
    #     # col="threshold",
    #     # row="metric_name",
    #     palette=palette,
    #     height=5,
    #     # aspect=0.75,
    #     # facet_kws=dict(sharex=False),
    #     kind="line",
    #     # legend="brief",
    #     # legend="out",
    #     data=df,
    # )
    # g = seaborn.scatterplot(
    #     x="Recall",
    #     y="Precision",
    #     hue="network",
    #     size="threshold",
    #     # col="threshold",
    #     # row="metric_name",
    #     palette=palette,
    #     # height=5,
    #     # aspect=0.75,
    #     # facet_kws=dict(sharex=False),
    #     # kind="line",
    #     # legend="brief",
    #     # legend="out",
    #     data=df,
    # )
    df["abs(z)"] = numpy.abs(df["z"])
    # df = df[numpy.logical_and(df["threshold"] > 0.001, df["threshold"] < 0.5)]  # changed here
    # df = df["abs(z)"] < 40]

    z_mean_df = df.groupby(["network", "hyperparameter_name"], as_index=False).mean()

    txt_out_path = out_dir / f"{plot_name}z_mean.txt"
    z_mean_df.loc[:, ["Recall", "Precision", "network"]].dropna(thresh=2).to_csv(txt_out_path, sep="\t", index=False)

    # z mean
    g = seaborn.relplot(
        x="Recall",
        y="Precision",
        hue="network",
        # size="threshold",
        # style="z",
        # col="z",
        # col_wrap=8,
        # row="metric_name",
        palette=palette,
        height=4,
        aspect=1.08,
        # facet_kws=dict(sharex=False),
        # kind="line",
        # legend="brief",
        # legend="out",
        data=z_mean_df,
    )
    g.fig.axes[0].set_xlim(0, 1)
    g.fig.axes[0].set_ylim(0, 1)

    plt.savefig(out_dir / f"{plot_name}z_mean.png")
    plt.savefig(out_dir / f"{plot_name}z_mean.svg")

    for selected_threshold in [0.1, 0.05, 0.01, 0.005]:
        z_rolling_df = (
            df[df["threshold"] == selected_threshold]
            .groupby(["network", "hyperparameter_name"])
            .rolling(5, center=True)
            .mean()
            .reset_index()
        )
        txt_out_path = out_dir / f"{plot_name}_threshold={selected_threshold}.txt"
        z_rolling_df.loc[:, ["Recall", "Precision", "network"]].dropna(thresh=2).to_csv(txt_out_path, sep="\t", index=False)

        g = seaborn.relplot(
            x="Recall",
            y="Precision",
            hue="network",
            # size="threshold",
            # style="z",
            # col="z",
            # col_wrap=8,
            # row="metric_name",
            palette=palette,
            hue_order=hue_order,
            height=4,
            # aspect=0.75,
            # facet_kws=dict(sharex=False),
            # kind="line",
            # legend="brief",
            # legend="out",
            data=z_rolling_df,
        )
        g.axes[0, 0].set_title(f"threshold={selected_threshold}")
        g.fig.axes[0].set_xlim(0, 1)
        g.fig.axes[0].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(out_dir / f"{plot_name}_threshold={selected_threshold}.png")
        plt.savefig(out_dir / f"{plot_name}_threshold={selected_threshold}.svg")

    # for ax in g.axes:
    #     print(ax)
    #
    # g.set_titles("{col_name}")

    # return axes[len(metrics) :]
    return g


if __name__ == "__main__":
    seaborn.set_context("poster")
    # parser = ArgumentParser(description="plot metrics")
    # args = parser.parse_args()
    #

    # fig, axes = plt.subplots(4, 2, figsize=(7, 12))
    # fig, axes = plt.subplots(4, 2, figsize=(6, 10.5))
    # fig, axes = plt.subplots(2, 2, figsize=(6, 5))
    # fig, axes = plt.subplots(4, 1, figsize=(3, 10))
    # fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    # fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    # if isinstance(axes, numpy.ndarray):
    #     axes = axes.flatten()
    # else:
    #     axes = [axes]

    out_dir = Path("/Users/fbeut/Desktop/lnet_stuff/plots/figs/pr")
    out_dir.mkdir(exist_ok=True, parents=True)
    plot_name = ""
    # axes = add_plot("heart_static_1", axes)
    # plot_name += "heart_static_1"
    # axes = add_plot("heart_dynamic_1", axes)
    # plot_name += "heart_dynamic_1"
    # axes = add_plot("beads", axes)
    # plot_name += "beads"
    axes = add_plot("beads_pr", [], out_dir)
    plot_name += "beads_pr"

    # plt.tight_layout()
    # plt.show()
