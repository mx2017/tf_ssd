if [[ "$1" == "squeezeDet" ]]
then
  python ./src/train.py \
  --dataset=KITTI \
  --data_path=./data/KITTI \
  --image_set=train \
  --train_dir=/home/disk3/tf_workspace/squeezeDet-master/logs/squeezedet/train \
  --net=squeezeDet \
  --summary_step=100 \
  --checkpoint_step=50 \
  --gpu=$USE_GPU
fi