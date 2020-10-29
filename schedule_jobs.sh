#sbatch run_hylfm.sh configs/train/heart/z_out49/static_f4.yml --checkpoint /g/kreshuk/beuttenm/pycharm_tmp/repos/hylfm-net/logs/train/beads/z_out49/small_f4/20-10-28_16-14-56/train/run000/checkpoints/v1_checkpoint_47_smooth_l1_loss-scaled\=-0.00013791094352200162.pth

sbatch run_hylfm.sh configs/train/heart/z_out49/bad_static_f4.yml
