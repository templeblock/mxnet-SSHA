# mxnet-SSHA

Original Caffe code: [https://github.com/mahyarnajibi/SSH](https://github.com/mahyarnajibi/SSH)

Deepinsight SSH Mxnet code: [https://github.com/deepinsight/mxnet-SSH](https://github.com/deepinsight/mxnet-SSH)

Evaluation on WIDER FACE:

| Impelmentation     | Easy-Set | Medium-Set | Hard-Set |
| ------------------ | -------- | ---------- | -------- |
| Original Caffe SSH | 0.93123  | 0.92106    | 0.84582  |
| Deepinsight SSH Model      | 0.93489  | 0.92281    | 0.84525  |
| Our SSHA Model      | -  | -    | -  |

### Installation
1. Clone the repository.

2. Download MXNet VGG16 pretrained model from [here](http://data.dmlc.ml/models/imagenet/vgg/vgg16-0000.params) and put it under `model` directory.

3. Type `make` to build necessary cxx libs.

### traindata
1. `mkdir data`
2. `cd data` `ln -s $(widerface dir) widerface`
3. You should make keypoint label 'wider_face_train_bbx_kp_gt.txt' instead of 'wider_face_train_bbox_gt.txt'

### Training

```
python train_ssh.py
```

### Test
[SSHA](https://github.com/ElegantGod/SSHA)
