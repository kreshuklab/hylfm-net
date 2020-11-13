#sbatch run_hylfm.sh configs/train/heart/z_out49/static_f4.yml --checkpoint /g/kreshuk/beuttenm/pycharm_tmp/repos/hylfm-net/logs/train/beads/z_out49/small_f4/20-10-28_16-14-56/train/run000/checkpoints/v1_checkpoint_47_smooth_l1_loss-scaled\=-0.00013791094352200162.pth

#sbatch run_hylfm.sh configs/train/heart/z_out49/refine_from_medium_beads.yml
#sbatch run_hylfm.sh configs/train/beads/z_out49/medium_lr_f4.yml

#sbatch run_hylfm.sh configs/test/heart/z_out49/contin_validate_f4.yml --checkpoint /g/kreshuk/LF_computed/lnet/logs/heart2/z_out49/static_f4_b2_with5_pois/20-06-13_20-26-33/train2/run000/checkpoints/v1_checkpoint_498_MS_SSIM=0.9710696664723483.pth

#sbatch run_hylfm.sh configs/train/heart/z_out49/refine_from_static_heart.yml
#sbatch run_hylfm.sh configs/train/heart/z_out49/refine_from_mednlarge_beads.yml
#sbatch run_hylfm.sh configs/train/heart/z_out49/refine_from_bad_static_heart.yml
#sbatch run_hylfm.sh configs/train/heart/z_out49/refine_from_medium_lr_beads.yml

#sbatch run_hylfm.sh configs/train/heart/refine_fish2/from_bad_static_heart.yml
#sbatch run_hylfm.sh configs/train/heart/refine_fish2/from_heart_lr.yml
#sbatch run_hylfm.sh configs/train/heart/refine_fish2/from_medium_beads.yml
#sbatch run_hylfm.sh configs/train/heart/refine_fish2/from_mednlarge_beads.yml
#sbatch run_hylfm.sh configs/train/heart/refine_fish2/from_static_heart.yml
#sbatch run_hylfm.sh configs/train/heart/refine_fish2/from_medium_beads_lr.yml  # not yet scheduled

#sbatch run_hylfm.sh configs/train/heart/refine_fish2/from_medium_beads_lr.yml --checkpoint /g/kreshuk/beuttenm/pycharm_tmp/repos/hylfm-net/logs/train/beads/z_out49/medium_f4/20-11-10_18-23-42/train/run000/checkpoints/v1_checkpoint_82_smooth_l1_loss-scaled\=-3.11052703182213e-05.pth
#sbatch run_hylfm.sh configs/test/heart/validate_fish2/from_static_heart.yml



# refine fish2 full pred
sbatch run_hylfm.sh configs/train/heart/refine_fish2_fullpred/from_static_heart.yml
sbatch run_hylfm.sh configs/train/heart/refine_fish2_fullpred/from_heart_lr.yml
sbatch run_hylfm.sh configs/train/heart/refine_fish2_fullpred/from_medium_beads.yml
#sbatch run_hylfm.sh configs/train/heart/refine_fish2_fullpred/from_bad_static_heart.yml
#sbatch run_hylfm.sh configs/train/heart/refine_fish2_fullpred/from_mednlarge_beads.yml
