torchrun --nnodes=1 \
    --nproc_per_node=1 \
    --master_port=12565 \
    ./computing_cost.py -p ./options/cost/Runtime.yml 