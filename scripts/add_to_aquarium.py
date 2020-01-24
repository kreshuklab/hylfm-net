import argparse
from pathlib import Path


def link_to_aquarium(tgt: Path):
    fish_tank = "/g/kreshuk/beuttenm/repos/lnet/logs/fish/"
    assert fish_tank in tgt.as_posix()
    assert tgt.exists()
    aquarium = Path(tgt.as_posix().replace(fish_tank, "/g/kreshuk/beuttenm/repos/lnet/logs/aquarium/"))
    aquarium.parent.mkdir(exist_ok=True, parents=True)
    aquarium.symlink_to(tgt)
    print(aquarium)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fish", nargs="+")

    args = parser.parse_args()

    [link_to_aquarium(Path(fish).resolve()) for fish in args.fish]


"""
20191208_tight/19-12-16_17-19-01 20191208_tight/19-12-18_22-18-43 20191208_tight/20-01-13_10-10-01 20191208_tight/20-01-13_10-34-01 20191208_tight/20-01-13_10-48-52
"""
