from hylfm.hylfm_types import CriterionChoice
from hylfm import criteria


def test_crit_choice():
    for key in CriterionChoice:
        name = key.name  # noqa
        assert hasattr(criteria, name)
