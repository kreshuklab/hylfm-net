# sbatch run_lnet.sh experiment_configs/brain1/test_z_out49/f8.yml --test --delete_existing_log_folder --checkpoint /g/kreshuk/LF_computed/lnet/logs/brain1/z_out49/f8_only11_2/20-05-20_03-28-05/train/run000/checkpoints/v1_checkpoint_*.pth
# sbatch run_lnet.sh experiment_configs/brain1/test_z_out49/f8.yml --test --delete_existing_log_folder --checkpoint /g/kreshuk/LF_computed/lnet/logs/brain1/z_out49/f8_bm4_with11_2/20-05-23_14-05-26/train/run000/checkpoints/v1_checkpoint_*.pth
# sbatch run_lnet.sh experiment_configs/brain1/test_z_out49/f8.yml --test --delete_existing_log_folder --checkpoint /g/kreshuk/LF_computed/lnet/logs/brain1/z_out49/f8f4_bm4_with11_2/20-05-23_14-10-16/train/run000/checkpoints/v1_checkpoint_*.pth
# sbatch run_lnet.sh experiment_configs/brain1/test_z_out49/f8.yml --test --delete_existing_log_folder --checkpoint /g/kreshuk/LF_computed/lnet/logs/brain1/z_out49/f8_bm4_with11_2_noSP/20-05-27_15-18-02/train/run000/checkpoints/v1_checkpoint_*.pth

# sbatch run_lnet.sh experiment_configs/brain1/test_z_out49/f4.yml --test --delete_existing_log_folder --checkpoint /g/kreshuk/LF_computed/lnet/logs/brain1/z_out49/f4_b2_only09_3/20-05-30_10-41-55/train/run000/checkpoints/v1_checkpoint_*.pth
# sbatch run_lnet.sh experiment_configs/brain1/test_z_out49/f4.yml --test --delete_existing_log_folder --checkpoint /g/kreshuk/LF_computed/lnet/logs/brain1/z_out49/f4f2_with11_2/20-05-19_12-29-44/train/run000/checkpoints/v1_checkpoint_*.pth
# sbatch run_lnet.sh experiment_configs/brain1/test_z_out49/f4.yml --test --delete_existing_log_folder --checkpoint /g/kreshuk/LF_computed/lnet/logs/brain1/z_out49/f4_b2_bm2_with11_2/20-05-23_14-13-18/train/run000/checkpoints/v1_checkpoint_*.pth

# sbatch run_lnet.sh experiment_configs/brain1/test_z_out49/f4.yml --test --delete_existing_log_folder --checkpoint /g/kreshuk/LF_computed/lnet/logs/brain1/z_out49/f4f2_only11_2/20-05-19_12-30-26/train/run000/checkpoints/v1_checkpoint_*.pth
# sbatch run_lnet.sh experiment_configs/brain1/test_z_out49/f4.yml --test --delete_existing_log_folder --checkpoint /g/kreshuk/LF_computed/lnet/logs/brain1/z_out49/f4_only11_2/20-05-18_20-28-48/train/run000/checkpoints/v1_checkpoint_*.pth

# sbatch run_lnet.sh experiment_configs/brain1/test_z_out49/f4.yml --test --delete_existing_log_folder --checkpoint /g/kreshuk/LF_computed/lnet/logs/brain1/z_out49/f4_only11_2/20-05-18_20-28-48/train/run000/checkpoints/v1_checkpoint_*.pth


# python -m lnet --cuda 3 experiment_configs/brain1/test_z_out49/f4.yml --test --checkpoint /g/kreshuk/LF_computed/lnet/logs/brain2/z_out49/f4_b2_bm2_with11_2_noSP/20-06-18_16-13-38/train2/run000/checkpoints/v1_checkpoint_2000_MS_SSIM\=0.6581869482994079.pth

# python -m lnet --cuda 3 experiment_configs/brain1/test_z_out49/f4.yml --test --delete_existing_log_folder --checkpoint /g/kreshuk/LF_computed/lnet/logs/brain1/z_out49/f4_b2_only11_2/20-06-06_17-59-42/train/run000/checkpoints/v1_checkpoint_37500_MS_SSIM\=0.8822072593371073.pth

# python -m lnet --cuda 3 experiment_configs/brain1/test_z_out49/f4.yml --test --checkpoint /g/kreshuk/LF_computed/lnet/logs/brain1/z_out49/f4_b2_only11_2/20-06-06_17-59-42/train/run000/checkpoints/v1_checkpoint_37500_MS_SSIM\=0.8822072593371073.pth
# python -m lnet --cuda 3 experiment_configs/brain1/test_z_out49/f4.yml --test --checkpoint /g/kreshuk/LF_computed/lnet/logs/brain1/z_out49/f4_b2_only09_3/20-05-30_10-41-55/train/run000/checkpoints/v1_checkpoint_82000_MSSSIM\=0.8523718668864324.pth

# python -m lnet --cuda 3 experiment_configs/brain2/test_z_out49/f4_16ms.yml --delete_existing_log_folder --test --checkpoint /g/kreshuk/LF_computed/lnet/logs/brain1/z_out49/f4_b2_only11_2/20-06-06_17-59-42/train/run000/checkpoints/v1_checkpoint_37500_MS_SSIM\=0.8822072593371073.pth
# python -m lnet --cuda 3 experiment_configs/brain2/test_z_out49/f4_20Hz_on_train.yml --delete_existing_log_folder --test --checkpoint /g/kreshuk/LF_computed/lnet/logs/brain1/z_out49/f4_b2_only11_2/20-06-06_17-59-42/train/run000/checkpoints/v1_checkpoint_37500_MS_SSIM\=0.8822072593371073.pth
# python -m lnet --cuda 3 experiment_configs/brain2/test_z_out49/f4_test_on_09_1.yml --delete_existing_log_folder --test --checkpoint /g/kreshuk/LF_computed/lnet/logs/brain1/z_out49/f4_b2_only11_2/20-06-06_17-59-42/train/run000/checkpoints/v1_checkpoint_37500_MS_SSIM\=0.8822072593371073.pth
# python -m lnet --cuda 3 experiment_configs/brain2/test_z_out49/f4_test_on_09_1_ls.yml --delete_existing_log_folder --test --checkpoint /g/kreshuk/LF_computed/lnet/logs/brain1/z_out49/f4_b2_only11_2/20-06-06_17-59-42/train/run000/checkpoints/v1_checkpoint_37500_MS_SSIM\=0.8822072593371073.pth

