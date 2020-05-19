from pathlib import Path

slurm_out_path = Path("/scratch/beuttenm/lnet/slurm_out")
slurm_out_prep_path = Path("/scratch/beuttenm/lnet/slurm_out_prep")

if __name__ == "__main__":
    job_id = "61075088_175"
    for outpath in slurm_out_path.glob(f"*{job_id}*.out"):
        print("out")
        print(outpath.read_text())
        errpath = outpath.with_suffix(".err")
        assert errpath.exists(), errpath
        print("err")
        print(errpath.read_text())

    for outpath in slurm_out_prep_path.glob(f"*{job_id}*.out"):
        print("out")
        print(outpath.read_text())
        errpath = outpath.with_suffix(".err")
        assert errpath.exists(), errpath
        print("err")
        print(errpath.read_text())