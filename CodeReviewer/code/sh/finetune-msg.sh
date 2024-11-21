# batch size 6 for 16 GB GPU

#mnt_dir="/home/codereview"
mnt_dir="/u/student/2022/cs22mds15020/code/code_review_download/CodeReviewer"

# You may change the following block for multiple gpu training
MASTER_HOST=localhost && echo MASTER_HOST: ${MASTER_HOST}
MASTER_PORT=23333 && echo MASTER_PORT: ${MASTER_PORT}
RANK=0 && echo RANK: ${RANK}
PER_NODE_GPU=1 && echo PER_NODE_GPU: ${PER_NODE_GPU}
WORLD_SIZE=1 && echo WORLD_SIZE: ${WORLD_SIZE}
NODES=1 && echo NODES: ${NODES}
NCCL_DEBUG=INFO
TORCH_DISTRIBUTED_DEBUG=DETAIL

bash test_nltk.sh


# Change the arguments as required:
#   model_name_or_path, load_model_path: the path of the model to be finetuned
#   eval_file: the path of the evaluation data
#   output_dir: the directory to save finetuned model (not used at infer/test time)
#   out_file: the path of the output file
#   train_filename: can be a directory contraining files named with "train*.jsonl"
#   raw_input: to select the preprocess method, set to True in this task

CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=${PER_NODE_GPU} --nnodes=${NODES} --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_HOST}:${MASTER_PORT} ../run_finetune_msg.py  \
  --model_type llama \
  --config_name meta-llama/Llama-2-7b-chat-hf \
  --add_lang_ids \
  --train_epochs 1 \
  --output_dir ../../save/gen_llama_10 \
  --train_filename ${mnt_dir}/dataset/Comment_Generation/msg-train.jsonl \
  --dev_filename ${mnt_dir}/dataset/Comment_Generation/msg-valid.jsonl \
  --max_source_length 300 \
  --max_target_length 128 \
  --train_batch_size 6 \
  --learning_rate 3e-4 \
  --gradient_accumulation_steps 3 \
  --mask_rate 0.15 \
  --save_steps 10 \
  --log_steps 10\
  --train_steps 10 \
  --gpu_per_node=${PER_NODE_GPU} \
  --node_index=${RANK} \
  --seed 2233 \
  --raw_input \

#python -m torch.distributed.launch --nproc_per_node ${PER_NODE_GPU} --node_rank=${RANK} --nnodes=${NODES} --master_addr=${MASTER_HOST} --master_port=${MASTER_PORT} ../run_finetune_msg.py  \
#  --train_epochs 3 \
#  --model_name_or_path microsoft/codereviewer \
#  --output_dir ../../save/gen \
#  --train_filename ../../dataset/gen-train.jsonl \
#  --dev_filename ../../dataset/gen-valid.jsonl \
#  --max_source_length 512 \
#  --max_target_length 128 \
#  --train_batch_size 6 \
#  --learning_rate 3e-4 \
#  --gradient_accumulation_steps 3 \
#  --mask_rate 0.15 \
#  --save_steps 10 \
#  --log_steps 10 \
#  --train_steps 10 \
#  --gpu_per_node=${PER_NODE_GPU} \
#  --node_index=${RANK} \
#  --seed 2233 \
#  --raw_input 
