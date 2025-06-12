torchrun --nnodes=1 \
    --nproc_per_node=1 \
    --master_port=12565 \
    ./train.py -p ./options/train/RSBlur.yml

