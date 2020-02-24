import argparse
import subprocess
from pathlib import Path

import yaml

if __name__ == "__main__":
    lnet_path = Path("/g/kreshuk/beuttenm/repos/lnet")

    parser = argparse.ArgumentParser()
    parser.add_argument("comp", type=str, help="e.g. comp/fdyn1")
    parser.add_argument("model_group", type=str, help="e.g. fish")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--cuda", type=int, default=3)

    args = parser.parse_args()
    prediction_only = args.comp.endswith("prediction")
    model_group = args.model_group
    model_group_yml = lnet_path / "trained_models" / f"{model_group}.yml"
    assert model_group_yml.exists()
    comp_template_yml = lnet_path / "lnet/experiment_configs" / f"{args.comp}.yml"
    assert comp_template_yml.exists()
    start = args.start
    cuda = args.cuda

    with model_group_yml.open() as f:
        trained_models = sorted(list(yaml.safe_load(f).items()))

    for i in range(3):
        print(i, trained_models[i])

    with comp_template_yml.open() as f:
        comp_template = yaml.safe_load(f)

    def compatible(trained):
        return (
            trained["model"]["nnum"]
            == comp_template["model"]["nnum"]
            # and trained["model"]["z_out"] == comp_template["model"]["z_out"]
        )

    for i, (tm, tm_config) in enumerate(trained_models[start:]):
        i += start
        if not compatible(tm_config):
            print("skipping", i, tm)
            continue

        print("evaluating", i, tm)
        tm_config["log"] = comp_template["log"]
        if "kwargs" in tm_config["model"] and "affine_transform_classes" in tm_config["model"]["kwargs"]:
            tm_config["model"]["kwargs"]["affine_transform_classes"] = comp_template["model"]["kwargs"][
                "affine_transform_classes"
            ]

        tm_config["eval"].update(comp_template["eval"])

        print("batch size", tm_config["eval"]["batch_size"])

        if prediction_only:
            tm_config["eval"]["transforms"] = [
                trf
                for trf in tm_config["eval"]["transforms"]
                if not isinstance(trf, dict) or trf.get("kwargs", {}).get("apply_to", 0) != 1
            ]

        tm_id = Path(tm).parent.name
        tm_yml = (comp_template_yml.parent / tm_id).with_suffix(".yml")
        with tm_yml.open("w") as f:
            yaml.safe_dump(tm_config, f)

        cmd = ["python", "-m", "lnet", "--cuda", str(cuda), str(tm_yml)]
        print(" ".join(cmd))
        subprocess.run(cmd, shell=False, check=True)
        tm_yml.unlink()

    print("last:", i)
