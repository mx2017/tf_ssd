CAFFE_MODEL=./data/checkpoints/VGG_ILSVRC_16_layers_fc_reduced.caffemodel
python tools/caffe_to_tensorflow.py \
    --model_name=vgg16 \
    --num_classes=21 \
    --caffemodel_path=${CAFFE_MODEL}