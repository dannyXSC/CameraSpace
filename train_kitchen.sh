name="kitchen_d0"
task_name="Kitchen_D0"
dataset_path="/tmp/core_datasets/kitchen/demo_src_kitchen_task_D0/demo.hdf5"

# Create a directory for logs based on the name
log_dir="log/$name"
mkdir -p "$log_dir"

#########################
# new template with nohup
#########################

# Run the first Python training script in the background with nohup, outputting to log/${name}/output1.log
python train.py --config-dir=./diffusion_policy/config \
    --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
    training.seed=42 training.device=cuda:5 \
    task.name="$name" task.task_name="$task_name" \
    task.dataset_path="$dataset_path" \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

# # Run the second Python training script in the background with nohup, outputting to log/${name}/output2.log
# nohup python train.py --config-dir=./diffusion_policy/config \
#     --config-name=train_diffusion_transformer_hybrid_workspace.yaml \
#     task=mimicgen_cs_hybrid_template \
#     training.seed=42 training.device=cuda:6 \
#     task.name="${name}_cs" task.task_name="$task_name" \
#     task.dataset_path="$dataset_path" \
#     hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}' > "$log_dir/output2.log" 2>&1 &

# Wait for all background jobs to finish
# wait
# echo "All tasks completed!"
