from hylfm.hylfm_types import Array, DatasetName, DatasetPart, TransformLike


class MetricsPipeline:
    def __init__(self, dataset_name: DatasetName, dataset_part: DatasetPart):
        if dataset_name in [DatasetName.beads_sample0, DatasetName.beads_highc_a]:
            self.groups = "beads"
            self.tgt = "ls_reg"
        elif dataset_name in [DatasetName.heart_static_a]:
            self.groups = "heart"
            self.tgt = "ls_trf"
        else:
            raise NotImplementedError(dataset_name)

        self.metrics = []


    def __call__(self, *, pred: Array, **other):
        tgt = other[self.tgt]
