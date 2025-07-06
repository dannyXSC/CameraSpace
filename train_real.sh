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


python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_unet_image_total.yaml \
    task=519_img \
    training.device="cuda:4" \
    training.checkpoint_every=50 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'


python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_unet_image_total.yaml \
    task=h2r_grab \
    policy.robot_as_condition=1 \
    training.device="cuda:2" \
    training.checkpoint_every=50 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_robot_as_condition'

python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_unet_image_total.yaml \
    task=h2r_grab \
    policy.robot_as_condition=2 \
    training.device="cuda:1" \
    training.checkpoint_every=50 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_human_as_condition'

python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_unet_image_total.yaml \
    task=h2r_grab \
    policy.robot_as_condition=0 \
    training.device="cuda:0" \
    training.checkpoint_every=50 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_only_obs'