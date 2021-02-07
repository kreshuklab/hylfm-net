from hylfm.hylfm_types import Array, DatasetChoice, DatasetPart, TransformLike


class MetricsPipeline:
    def __init__(self, dataset_name: DatasetChoice, dataset_part: DatasetPart):
        if dataset_name in [DatasetChoice.beads_sample0, DatasetChoice.beads_highc_a]:
            self.groups = "beads"
            self.tgt = "ls_reg"
        elif dataset_name in [DatasetChoice.heart_static_a]:
            self.groups = "heart"
            self.tgt = "ls_trf"
        else:
            raise NotImplementedError(dataset_name)

        self.metrics = []


    def __call__(self, *, pred: Array, **other):
        tgt = other[self.tgt]
