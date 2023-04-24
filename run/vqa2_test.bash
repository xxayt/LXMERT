# The name of this experiment.
name=$2

# Save logs and models under logs/vqa; make backup.
output=logs/vqa/$name

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/vqa_test.py \
    --tiny --train train --valid ""  \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --batchSize 64 --optim bert --lr 5e-5 --epochs 10 \
    --tqdm --name $name --save_dir $output ${@:3}
