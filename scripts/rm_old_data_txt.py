from pathlib import Path

data_path = Path("../data")


for txt in data_path.glob("*.txt"):
    n5 = txt.with_suffix(".n5")
    if not n5.exists():
        txt.unlink()
