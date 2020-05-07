from lnet.datasets.gcamp import g200311_083021_ls

from lnet.datasets import N5CachedDatasetFromInfoSubset, get_dataset_from_info
import matplotlib.pyplot as plt

if __name__ == "__main__":

    info = g200311_083021_ls
    info.transformations += [
        {
            "Resize": {
                "apply_to": "ls",
                "shape": [1.0, 1.0, 0.21052631578947368421052631578947, 0.21052631578947368421052631578947],
                "order": 2,
            }
        },  # 2/19=0.10526315789473684210526315789474; 4/19=0.21052631578947368421052631578947; 8/19=0.42105263157894736842105263157895
        {"Assert": {"apply_to": "ls", "expected_tensor_shape": [None, 1, 1, None, None]}},
    ]
    test_ls_slice_dataset = get_dataset_from_info(info=info, cache=True)

    for p in [2]:
        ds = N5CachedDatasetFromInfoSubset(
            dataset=test_ls_slice_dataset,
            indices=None,
            filters=[("instensity_range", {"apply_to": "ls", "max_above": {"mean+xstd": p}})],
        )
        print(p, len(ds))

        min_img_max = 999999
        for s in ds:
            img = s["ls"]
            assert img.shape[0] == 1
            assert img.shape[1] == 1
            assert img.shape[2] == 1
            img = img[0, 0, 0]
            max_ = img.max()
            if max_ < min_img_max:
                min_img = img
                min_img_max = max_
                plt.imshow(img)
                plt.title(f"p: {p}, max: {max_}")
                plt.show()
