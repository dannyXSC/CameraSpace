python eval_real_test.py \
-i /home/ubuntu/danny/diffusion_policy/data/outputs/2025.04.13/11.42.51_train_diffusion_unet_hybrid_square_image/checkpoints/latest.ckpt \
-t /home/ubuntu/project/Moore-AnimateAnyone/data/writing/writing_cvpr/episode_4.hdf5 --device cuda:0

python eval_real_common.py \
-i /home/ubuntu/danny/diffusion_policy/data/outputs/2025.05.12/07.30.23_train_diffusion_unet_hybrid_h2r_cloth/checkpoints/epoch=0550-train_loss=0.031.ckpt \
-t /data1/dataset/cloth/eval/cloth12.hdf5 --device cuda:7

python eval_real_common.py \
-i /home/ubuntu/danny/diffusion_policy/data/outputs/2025.05.12/07.55.14_train_diffusion_unet_hybrid_h2r_push_box/checkpoints/epoch=0150-train_loss=0.008.ckpt \
-t /data1/dataset/push_box_common_v1/episode_0.hdf5 --device cuda:7

