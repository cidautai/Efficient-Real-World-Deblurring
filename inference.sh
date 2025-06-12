torchrun --nnodes=1 \
    --nproc_per_node=1 \
    --master_port=12565 \
    ./inference.py -p ./options/inference/RSBlur.yml -c ./models/NAFNet-C16-L28_RSBlur.pt -i ./data/datasets/RSBlur/development_input_RSBlur
