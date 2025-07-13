name="mug_256_mask"
task_name="MugCleanup_D0"
dataset_path="/data/mimicgen/core_datasets_512/mug_cleanup/foreground/D0.hdf5"

# Create a directory for logs based on the name
log_dir="log/$name"
mkdir -p "$log_dir"

#########################
# new template with nohup
#########################

# Run the first Python training script in the background with nohup, outputting to log/${name}/output1.log
# python train.py --config-dir=./diffusion_policy/config \
#     --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
#     training.seed=42 training.device=cuda:0 \
#     task=mimicgen_cs_mask_wpose_template \
#     task.dataset_path="$dataset_path" \
#     task.name="$name" \
#     task.task_name="$task_name" \
#     task.if_mask=True \
#     hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_mask_cs' \
#     logging.name='${now:%Y.%m.%d-%H.%M.%S}_mask_cs'

python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_mimicgen_256.yaml \
    training.seed=42 training.device=cuda:1 \
    task=mimicgen_hybrid_256_template \
    task.dataset_path="$dataset_path" \
    task.name="$name" \
    task.task_name="$task_name" \
    task.if_mask=True \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_mask' \
    logging.name='${now:%Y.%m.%d-%H.%M.%S}_mask'

# Wait for all background jobs to finish
# wait
echo "All tasks completed!"
