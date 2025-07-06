# mix
python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
    task=mix_cs_hybrid_wpose_multiposition \
    training.device=cuda:4 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_mix'

python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
    task=mix_hybrid_multiposition \
    training.device=cuda:5 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_mix'

python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
    task=hammer_cs_hybrid_wpose_multiposition \
    training.device=cuda:0 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_cs'

python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
    task=hammer_hybrid_multiposition \
    training.device=cuda:1 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
    task=kitchen_cs_hybrid_wpose_multiposition \
    training.device=cuda:2 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_cs'

python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
    task=kitchen_hybrid_multiposition \
    training.device=cuda:3 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
    task=3piece_cs_hybrid_wpose_multiposition \
    training.device=cuda:4 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_cs'

python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
    task=3piece_hybrid_multiposition \
    training.device=cuda:5 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
    task=mug_cs_hybrid_wpose_multiposition \
    training.device=cuda:6 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_cs'

python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
    task=mug_hybrid_multiposition \
    training.device=cuda:7 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# mask
python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
    task=mug_cs_hybrid_wpose_multiposition_mask \
    task.if_mask=True \
    training.device=cuda:3 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_cs'

python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
    task=mug_hybrid_multiposition_mask \
    task.if_mask=True \
    training.device=cuda:2 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'


python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace_low.yaml \
    task=mug_cs_hybrid_wpose_multiposition \
    training.device=cuda:6 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_cs'

python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace_low.yaml \
    task=mug_hybrid_multiposition \
    training.device=cuda:7 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
    task=mug_cs_hybrid_wop_multiposition \
    training.device=cuda:0 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_cs'

python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
    task=mug_hybrid_wop_multiposition \
    training.device=cuda:1 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'


python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
    task=square_cs_hybrid_wpose_multiposition \
    training.device=cuda:0 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_cs'

python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
    task=square_hybrid_multiposition \
    training.device=cuda:1 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'   

python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
    task=stack3_cs_hybrid_wpose_multiposition \
    training.device=cuda:2 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_cs'

python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
    task=stack3_hybrid_multiposition \
    training.device=cuda:3 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'  

python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
    task=threading_cs_hybrid_wpose_multiposition \
    training.device=cuda:4 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_cs'

python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
    task=threading_hybrid_multiposition \
    training.device=cuda:5 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'  

python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
    task=coffee_cs_hybrid_wpose_multiposition \
    training.device=cuda:6 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_cs'

python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
    task=coffee_hybrid_multiposition \
    training.device=cuda:7 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'  

python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
    task=hammer_cs_hybrid_wop_wpose_multiposition \
    training.device=cuda:6 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}_cs'

python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
    task=hammer_hybrid_wop_multiposition \
    training.device=cuda:7 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
