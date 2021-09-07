# The name of this experiment.
name=$1

# Save logs and models
output=result/$name
mkdir -p $output

CUDA_VISIBLE_DEVICES=$2 python pretrain.py --epoch 7 --batch_size 64 --lr 1e-4 --output $output