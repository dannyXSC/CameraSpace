name="mug_d0"
task_name="MugCleanup_D0"
dataset_path="/home/ubuntu/danny/diffusion_policy/tests/my/data/cl_demos.hdf5"

# Create a directory for logs based on the name
log_dir="log/$name"
mkdir -p "$log_dir"

#########################
# new template with nohup
#########################

# Run the first Python training script in the background with nohup, outputting to log/${name}/output1.log
python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace_low.yaml \
    training.seed=42 training.device=cuda:0 \
    task=mug_hybrid_part_stage \
    task.max_samples=[0,800] \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${task.max_samples}_only_800_training' \
    logging.name='${now:%Y.%m.%d-%H.%M.%S}_800_part_training'

# python train.py --config-dir=./diffusion_policy/config \
#     --config-name=train_diffusion_transformer_hybrid_workspace_low.yaml \
#     training.seed=42 training.device=cuda:4 \
#     task=mug_hybrid_cl_contrast \
#     task.max_samples=[10] \
#     hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${task.max_samples}'

python train.py --config-dir=./diffusion_policy/config \
    --config-name=finetune_diffusion_transformer_hybrid_workspace.yaml \
    training.seed=42 training.device=cuda:6 \
    task=mug_hybrid_cl_fintune \
    task.max_samples=[0,800] \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${task.max_samples}_contrast'

python train.py --config-dir=./diffusion_policy/config \
    --config-name=finetune_diffusion_transformer_hybrid_workspace.yaml \
    training.seed=42 training.device=cuda:3 \
    task=mug_hybrid_cl_fintune \
    task.max_samples=[50] \
    task.dataset_path=[/tmp/core_datasets/mug_cleanup/demo_src_mug_cleanup_task_D0/demo.hdf5] \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${task.max_samples}_finetune'


# # Run the second Python training script in the background with nohup, outputting to log/${name}/output2.log
# nohup python train.py --config-dir=./diffusion_policy/config \
#     --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
#     task=mimicgen_cs_hybrid_template \
#     training.seed=42 training.device=cuda:0 \
#     task.name="${name}_cs" task.task_name="$task_name" \
#     task.dataset_path="$dataset_path" \
#     hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' > "$log_dir/output2.log" 2>&1 &



# Wait for all background jobs to finish
# wait
echo "All tasks completed!"
