# The name of this experiment.
name=$1

# Save logs and models
output=result/$name
mkdir -p $output

# train or test
CUDA_VISIBLE_DEVICES=$2 python train.py --epoch 15 --batch_size 24 --lr 3e-5 --load load/best.pth --dataset vqa2 --output $output --save_model