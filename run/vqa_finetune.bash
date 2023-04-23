# The name of this experiment.
name=$2

# Save logs and models under logs/vqa; make backup.
output=logs/vqa/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/vqa.py \
    --train train,nominival --valid minival  \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERTQA logs/pretrained/model \
    --batchSize 64 --optim bert --lr 5e-5 --epochs 10 \
    --tqdm --name $name --save_dir $output ${@:3}
