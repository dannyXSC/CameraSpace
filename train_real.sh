python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_unet_image_total.yaml \
    training.device="cuda:5" \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_unet_image_total.yaml \
    task=h2r_cloth \
    training.device="cuda:6" \
    training.checkpoint_every=50 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_unet_image_total.yaml \
    task=h2r_roll \
    training.device="cuda:4" \
    training.checkpoint_every=50 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_unet_image_total.yaml \
    task=h2r_push_box \
    training.device="cuda:4" \
    training.checkpoint_every=50 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'