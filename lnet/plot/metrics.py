# import seaborn as sns; sns.set()
# import matplotlib.pyplot as plt
# fmri = sns.load_dataset("fmri")
# ax = sns.lineplot(x="timepoint", y="signal", data=fmri)
# print(fmri)
#
# plt.show()

import typing
from argparse import ArgumentParser
from collections import OrderedDict
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy
import pandas
import seaborn
from ruamel.yaml import YAML

yaml = YAML(typ="safe")


def get_datasets(all_paths, scalar_data, metric_name):
    length = None
    lengths = None
    datasets = []
    per_vol = False
    for n, s, k in scalar_data:
        path = all_paths[n][metric_name][s]
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
                    lengths = [1 for mps in metric_per_sample]
                    # assert isinstance(metric_per_sample, list)
                else:
                    assert lengths == [len(mps) for mps in metric_per_sample]
                metric_per_sample = metric_per_sample
        else:
            assert length // z_slice_mod == length / z_slice_mod

        datasets.append(metric_per_sample)

    if z_slice_mod is None:
        if per_vol:
            z_slice = None
            z = None
        else:
            assert lengths is not None
            z_slice = numpy.concatenate([numpy.arange(l) for l in lengths])
            z = (z_slice - 49 / 2) * 5
    else:
        z_slice = numpy.tile(numpy.arange(z_slice_mod), length // z_slice_mod)
        z = (z_slice - 209 / 2) * 2

    return datasets, z_slice, z


if __name__ == "__main__":
    parser = ArgumentParser(description="plot metrics")
    # parser.add_argument("paths", nargs="+", type=Path)
    # parser.add_argument("--names", nargs="+", type=str)
    # parser.add_argument("--slice_mod", default=209, type=int)

    # args = parser.parse_args()
    #
    # assert len(args.paths) == len(args.names)

    metric_postfix = ""

    use_local_data = True
    if use_local_data:
        test_data_path = Path("C:/Users/fbeut/Desktop/lnet_stuff/plots/data/heart/static1")
    else:
        test_data_path = Path("K:/LF_computed/lnet/plain/heart/static1/test")
    z_slice_mod: typing.Optional[int] = None  # 209 None
    # care_result_path = Path("K:/LF_computed/lnet/care/results/heart")
    # hylfm_result_path = Path("K:/LF_computed/lnet/hylfm/results")
    # lr_result_path = Path("K:/LF_computed/lnet/hylfm/results")
    all_paths = {}
    for net in ["lr", "hylfm_dyn", "hylfm_stat", "v0_on_48x88x88"]:
        all_paths[net] = {}
        if use_local_data:
            result_path = test_data_path / net
        else:
            result_path = test_data_path / net / "metrics"

        all_paths[net] = {}
        for metric_name in ["PSNR", "NRMSE", "SSIM", "MS-SSIM", "MSE"]:
            all_paths[net][metric_name] = {}
            if metric_name in ["MSE"]:
                in_file_metric_name = metric_name + "_loss"
            else:
                in_file_metric_name = metric_name

            in_file_metric_name = in_file_metric_name.replace("-", "_")
            for scaled in [False, True]:
                ifmn = in_file_metric_name + metric_postfix
                if scaled:
                    ifmn += "-scaled"

                ifmn += "-along_z"
                file_name = f"{ifmn.lower()}.yml"
                all_paths[net][metric_name][scaled] = result_path / file_name

    # pprint(all_paths)

    better_network_names = {
        "hylfm_dyn": "HyLFM-Net dyn",
        "hylfm_stat": "HyLFM-Net stat",
        "v0_on_48x88x88": "CARE",
        "lr": "LFD",
    }
    palette = {
        "HyLFM-Net stat": "#648FFF",
        "HyLFM-Net dyn": "#785EF0",
        "CARE": "#DC267F",
        "LFD": "#FFB000",
    }  # blue: #648FFF, purple: #785EF0, red: #DC267F, dark orange: #FE6100, orange: #FFB000

    legend_loc = {"PSNR": "upper right", "NRMSE": "upper center", "SSIM": "lower left", "MS-SSIM": "lower left", "MSE": "upper left"}

    # fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
    fig, axes = plt.subplots(3, 1, figsize=(4, 10))
    axes = axes.flatten()

    plot_name = "fig2d"
    # metrics = ["PSNR", "MSE", "NRMSE", "SSIM", "MS-SSIM", None]
    metrics = ["PSNR", "SSIM", "MS-SSIM"]  #, "PSNR", "MSE", "NRMSE", "SSIM", "MS-SSIM", None]
    for i, metric_name in enumerate(metrics):
        ax = axes[i]
        if metric_name is None:
            ax.axis("off")
            continue

        scalar_columns = ["network", "scaled"]
        scalar_data = [
            ["hylfm_dyn", True],
            # ["hylfm_dyn", False],
            ["hylfm_stat", True],
            # ["hylfm_stat", False],
            ["lr", True],
            # ["lr", False],
            ["v0_on_48x88x88", True],
            # ["v0_on_48x88x88", False],
        ]
        title = ""  #  f"{net} {metric} on static heart"
        # plot_name = "_".join("".join(str(s) for s in sd) for sd in scalar_data)

        x_name = "z"
        map_labels = {}
        ############################
        scalar_columns.append("key")
        for i, dat in enumerate(scalar_data):
            dat.append(i)

        # print(scalar_df)

        datasets, z_slice, z = get_datasets(all_paths, scalar_data, metric_name)

        # value_dfs = [pandas.DataFrame(numpy.asarray([numpy.repeat([k], len(dat)), dat, z_slice, z]).T, columns=["key", metric_name, "z slice", "z"]) for k, dat in enumerate(datasets)]
        value_dfs = [
            pandas.DataFrame({"key": numpy.repeat([k], len(dat)), metric_name: dat, "z_slice": z_slice, "z": z})
            for k, dat in enumerate(datasets)
        ]
        # for vdf in value_dfs:
        #     print(vdf)

        for sd in scalar_data:
            sd[0] = better_network_names[sd[0]]
        scalar_df = pandas.DataFrame(scalar_data, columns=scalar_columns)
        df = scalar_df.merge(pandas.concat(value_dfs), on="key")
        # df = df.reset_index(drop=True)
        # print(df)
        # seaborn.set(style="ticks", color_codes=False)
        # seaborn.catplot(x="slice", y=first_name, kind="swarm", data=data)

        # .FacetGrid(att, col="subject", col_wrap=5, height=1.5)
        seaborn.lineplot(
            x=x_name,
            y=metric_name,
            data=df,
            hue="network",
            palette=palette,
            # style="scaled",
            # style_order=[True, False],
            legend="brief",
            ci="sd",
            ax=ax,
        )  # , legend="full", ax=ax)
        seaborn.despine(ax=ax)
        ax.legend(loc=legend_loc[metric_name])
        if x_name == "z slice":
            ax.set_xlim(0, z_slice.max())
        elif x_name == "z":
            ax.set(xlabel="z [μm]")
            ax.set_xlim(z.min(), z.max())

        # for name in metrics[slice(0, to_twin_at)]:
        # print(df)

        # if to_twin_at is not None:
        #     twinx = ax.twinx()
        #     for name in metrics[to_twin_at:]:
        #         seaborn.lineplot(x=x_name, y=name, data=df, legend="full", label=map_labels.get(name, name), ax=twinx)
        #
        #     twinx.set(ylabel=twin_y_label)

        ax.set_title(title)
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[1:], labels=labels[1:])
        # ax.set(ylabel=metric)
        # plt.legend()
        # plt.show()

    plt.tight_layout()
    out_dir = Path("/Users/fbeut/Desktop/lnet_stuff/plots/figs")
    out_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(out_dir / f"{plot_name.replace(' ', '_')}.png")
    fig.savefig(out_dir / "svgs" / f"{plot_name.replace(' ', '_')}.svg")
