# resave this file for editing as `settings.py`
experiment_configs_folder = "experiment_configs"

num_workers_train_data_loader: int = 0
num_workers_validate_data_loader: int = 0
num_workers_test_data_loader: int = 0
pin_memory: bool = True

max_workers_per_dataset = 8
reserved_workers_per_dataset_for_getitem = 8
max_workers_save_output: int = 16
max_workers_for_stat_per_ds: int = 16

assert reserved_workers_per_dataset_for_getitem <= max_workers_per_dataset
