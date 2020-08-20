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

yaml = YAML(typ="safe")


def get_datasets(all_paths, scalar_data, metric_name, z_slice_mod):
    length = None
    lengths = None
    datasets = []
    per_vol = False
    for net, hyperparameter_name, key in scalar_data:
        path = all_paths[net][metric_name][hyperparameter_name]
        metric_per_sample = yaml.load(path)
        if length is None:
            length = len(metric_per_sample)
        else:
            assert len(metric_per_sample) == length, (len(metric_per_sample), length)

        if z_slice_mod is None:
            if any(isinstance(mps, list) for mps in metric_per_sample):
                assert not per_vol
                assert all(isinstance(mps, list) for mps in metric_per_sample), [type(mps) for mps in metric_per_sample]
                if lengths is None:
                    lengths = [len(mps) for mps in metric_per_sample]
                    # assert isinstance(metric_per_sample, list)
                else:
                    assert lengths == [len(mps) for mps in metric_per_sample]

                metric_per_sample = numpy.concatenate(metric_per_sample)
            else:
                assert per_vol
                assert not any(isinstance(mps, list) for mps in metric_per_sample), [
                    type(mps) for mps in metric_per_sample
                ]
                if lengths is None:
                    lengths = [1] * len(metric_per_sample)
                    # assert isinstance(metric_per_sample, list)
                else:
                    assert lengths == [len(mps) for mps in metric_per_sample]
                metric_per_sample = metric_per_sample
        else:
            assert length // z_slice_mod == length / z_slice_mod

        # print(n)
        # print(metric_per_sample[length // 2 - 5: length//2+5])
        datasets.append(metric_per_sample)

    if z_slice_mod is None:
        if per_vol:
            z_slice = None
            z = None
        else:
            assert lengths is not None
            z_slice = numpy.concatenate([numpy.arange(l) for l in lengths])
            z = (z_slice - z_slice.max() // 2) * 5
            assert -z.min() == z.max()
    elif z_slice_mod == 51:
        z_slice = numpy.tile(numpy.arange(z_slice_mod), length // z_slice_mod)
        z = (z_slice - z_slice.max() // 2) * 2
        assert -z.min() == z.max()
    else:
        z_slice = numpy.tile(numpy.arange(z_slice_mod), length // z_slice_mod)
        z = -z_slice + z_slice.max() // 2
        assert -z.min() == z.max()

    return datasets, z_slice, z


def add_plot(plot_name, axes):
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
            scalar_data = [["hylfm", "scaled-True"], ["lr", "scaled-True"]]
            scalar_data.append(["v0_on_56x80x80", "scaled-True"])
        elif plot_name == "beads_pr":
            scalar_data = []
            for threshold in ["0.1", "0.05", "0.01", "0.001"]:
                hyper_parameter_name = f"distthreshold-3.0_excludeborder-False_maxsigma-6.0_minsigma-1.0_overlap-0.5_sigmaratio-3.0_threshold-{threshold}_scaled-True"
                scalar_data.append(["hylfm", hyper_parameter_name])
                scalar_data.append(["lr", hyper_parameter_name])
                scalar_data.append(["v0_on_56x80x80", hyper_parameter_name])
        else:
            raise NotImplementedError(plot_name)
        z_slice_mod: typing.Optional[int] = None  # 209 None
        nets = ["lr", "hylfm", "v0_on_56x80x80"]
        test_data_path = Path("K:/LF_computed/lnet/plain/beads/f8_01highc/test")
    else:
        raise NotImplementedError(plot_name)

    plot_name += metric_postfix + ""
    title = ""  #  f"{net} {metric} on static heart"
    x_name = "z"
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
            for file_path in result_path.glob(f"{metric_file_name_start}*.yml"):
                hyperparameter_name = file_path.stem[len(metric_file_name_start) :]
                if hyperparameter_name.endswith("-scaled"):
                    hyperparameter_name = hyperparameter_name.replace("-scaled", "_scaled-True")
                else:
                    hyperparameter_name += "_scaled-False"

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
    palette = {
        "HyLFM-Net": "#648FFF",
        "HyLFM-Net stat": "#648FFF",
        "HyLFM-Net dyn": "#785EF0",
        "LFD+CARE": "#DC267F",
        "LFD": "#FFB000",
    }  # blue: #648FFF, purple: #785EF0, red: #DC267F, dark orange: #FE6100, orange: #FFB000

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

    plot_name += "overview"
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
    dfs = []
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
                    "metric_name": metric_name,
                    "metric_value": dat,
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
            avg_values[metric_name][sd[0]] = vdf["metric_value"].mean(), vdf["metric_value"].std()

        print(metric_name)
        pprint(avg_values[metric_name])
        avg_values[metric_name] = None

        scalar_df = pandas.DataFrame(scalar_data_here, columns=scalar_columns_here)
        dfs.append(scalar_df.merge(pandas.concat(value_dfs), on="key"))
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

    df = pandas.concat(dfs)

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

    g = seaborn.relplot(
        x=x_name,
        y="metric_value",
        hue="network",
        # size="threshold",
        col="threshold",
        row="metric_name",
        palette=palette,
        height=5,
        # aspect=0.75,
        # facet_kws=dict(sharex=False),
        kind="line",
        # legend="brief",
        # legend="out",
        data=df,
    )
    # for ax in g.axes:
    #     print(ax)
    #
    # g.set_titles("{col_name}")

    # return axes[len(metrics) :]
    return g


if __name__ == "__main__":
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

    plot_name = ""
    # axes = add_plot("heart_static_1", axes)
    # plot_name += "heart_static_1"
    # axes = add_plot("heart_dynamic_1", axes)
    # plot_name += "heart_dynamic_1"
    # axes = add_plot("beads", axes)
    # plot_name += "beads"
    axes = add_plot("beads_pr", [])
    plot_name += "beads_pr"

    # plt.tight_layout()
    out_dir = Path("/Users/fbeut/Desktop/lnet_stuff/plots/figs")
    out_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(out_dir / f"{plot_name.replace(' ', '_')}.png")
    plt.savefig(out_dir / "svgs" / f"{plot_name.replace(' ', '_')}.svg")
    # plt.show()
