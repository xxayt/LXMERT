# The name of this experiment.
name=$2

# Save logs and models under logs/gqa; make backup.
output=logs/gqa/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/gqa.py \
    --train train,valid --valid testdev \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERTQA logs/pretrained/model \
    --batchSize 32 --optim bert --lr 1e-5 --epochs 4 \
    --tqdm --save_dir $output ${@:3}
