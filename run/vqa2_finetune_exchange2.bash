# The name of this experiment.
name=$2

# Save logs and models under logs/vqa; make backup.
output=logs/vqa/$name

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/vqa_train.py \
    --train train,nominival --valid minival  \
    --llayers 9 --xlayers 9 --rlayers 5 \
    --loadLXMERTQA logs/pretrained/model \
    --batchSize 64 --optim bert --lr 5e-5 --epochs 8 \
    --tqdm --name $name --save_dir $output ${@:3}
