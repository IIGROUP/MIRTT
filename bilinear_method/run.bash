# The name of this experiment: model_dataset
name=$1

# Save logs and models
output=result/$name
mkdir -p $output

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$2 python ./main.py --mod $3 --dataset $4 --model $5 --output $output ${@:6}