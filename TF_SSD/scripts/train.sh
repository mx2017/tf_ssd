#!/bin/bash

export USE_GPU=0

if [ $# -eq 0 ]
then
  echo "[ERROR] Missing an argument; please specify the model to use. Use -h or --help for more detailed usage instructions."
fi

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Usage: ./scripts/train.sh (squeezeDet|squeezeDet+|vgg16|resnet50) [options]"
      echo " "
      echo "options:"
      echo "-h, --help                show brief help"
      echo "-gpu                      run with gpu"
      exit 0
      ;;
    -gpu)
      shift
      export USE_GPU=1
      shift
      ;;
    *)
      break
      ;;
  esac
done

# =========================================================================== #
# command for squeezeDet:
# =========================================================================== #
#
if [[ "$1" == "squeezeDet" ]]
then
  python ./src/train.py \
  --dataset=KITTI \
  --pretrained_model_path=./data/SqueezeNet/squeezenet_v1.1.pkl \
  --data_path=./data/KITTI \
  --image_set=train \
  --train_dir=./logs/SqueezeDet/train \
  --net=squeezeDet \
  --summary_step=500 \
  --checkpoint_step=3000 \
  --restore_dir=./logs/SqueezeDet/model_save \
  --gpu=$USE_GPU \
  --max_steps=1000000
fi

#--restore_dir=./data/model_checkpoints/squeezeDet/ 
#--restore_dir=./logs/SqueezeDet/model_save

if [[ "$1" == "mobileDet" ]]
then
  echo "scripts:===>mobileDet"
  python ./src/train.py \
  --dataset=VKITTI \
  --data_path=./data/VKITTI \
  --image_set=train \
  --train_dir=./logs/MobileDet/train \
  --net=mobileDet \
  --summary_step=500 \
  --checkpoint_step=3000 \
  --gpu=$USE_GPU \
  --restore_dir=./logs/MobileDet/model_save \
  --max_steps=300000
fi

if [[ "$1" == "mobileDet_V1_025" ]]
then
  echo "scripts:===>mobileDet_V1_025"
  python ./src/train.py \
  --dataset=VKITTI \
  --data_path=./data/VKITTI \
  --image_set=train \
  --train_dir=./logs/MobileDet_V1_025/train \
  --net=mobileDet_V1_025 \
  --summary_step=500 \
  --checkpoint_step=3000 \
  --gpu=$USE_GPU \
  --restore_dir=./logs/MobileDet_V1_025/model_save \
  --max_steps=300000
fi


if [[ "$1" == "SSD" ]]
then
    echo "scripts:===>SSD"
    python ./src/train.py \
    --dataset=VKITTI \
    --data_path=./data/VKITTI \
    --image_set=train \
    --train_dir=./logs/SSD/train \
    --pretrained_model_path=./data/pretrain/VGG_VOC0712_SSD_300x300.pkl \
    --net=SSD \
    --summary_step=500 \
    --checkpoint_step=3000 \
    --gpu=$USE_GPU \
    --restore_dir=./logs/SSD/model_save \
    --max_steps=300000
fi

#--restore_dir=./data/MobileNet/
#--pretrained_model_path=./data/SqueezeNet/squeezenet_v1.1.pkl \
#--pretrained_model_path=./data/SqueezeNet/squeezenet_v1.1.pkl \
#--pretrained_model_path=./data/model_checkpoints/squeezeDet/model.ckpt-87000 
#

# =========================================================================== #
# command for squeezeDet+:
# =========================================================================== #
if [[ "$1" == "squeezeDet+" ]]
then
  python ./src/train.py \
  --dataset=KITTI \
  --pretrained_model_path=./data/SqueezeNet/squeezenet_v1.0_SR_0.750.pkl \
  --data_path=./data/KITTI \
  --image_set=train \
  --train_dir=/home./logs/SqueezeDetPlus/train \
  --net=squeezeDet+ \
  --summary_step=500 \
  --checkpoint_step=500 \
  --gpu=$USE_GPU
fi

if [[ "$1" == "vgg16" ]]
then
  python ./src/train.py \
  --dataset=VKITTI \
  --pretrained_model_path=./data/VGG16/VGG_ILSVRC_16_layers_weights.pkl \
  --data_path=./data/VKITTI \
  --image_set=train \
  --train_dir=/tmp/bichen/logs/vgg16/train \
  --net=vgg16 \
  --summary_step=500 \
  --checkpoint_step=500 \
  --gpu=$USE_GPU
fi

# =========================================================================== #
# command for resnet50:
# =========================================================================== #
if [[ "$1" == "resnet50" ]]
then
  python ./src/train.py \
  --dataset=KITTI \
  --pretrained_model_path=./data/ResNet/ResNet-50-weights.pkl \
  --data_path=./data/KITTI \
  --image_set=train \
  --train_dir=/tmp/bichen/logs/resnet/train \
  --net=resnet50 \
  --summary_step=100 \
  --checkpoint_step=500 \
  --gpu=$USE_GPU
fi