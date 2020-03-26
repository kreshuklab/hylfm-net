num_workers_train_data_loader: int = 16
num_workers_validate_data_loader: int = 16
num_workers_test_data_loader: int = 16
pin_memory: bool = True

max_workers_per_dataset = 8
reserved_workers_per_dataset_for_getitem = 8

assert reserved_workers_per_dataset_for_getitem <= max_workers_per_dataset
