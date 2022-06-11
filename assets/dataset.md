Data Preparation
===

1. Download the images from [HAKE](http://hake-mvig.cn/) dataset. You may follow the [official instruction](https://github.com/DirtyHarryLYL/HAKE/tree/master/Images#download-images-for-hake). For your convenience, you may download all the images required by Bongard-HOI [here](http://tensorlab.cms.caltech.edu/projects/bongard-hoi/data/bongard_hoi_images.tar). The images should be extracted to `./assets/data/hake/images` and the file structure looks like:
    ```plain
    data
    └── hake
        └── images
            ├── hake_images_20190730
            ├── hcvrd
            ├── hico_20160224_det
            │   └── images
            │       ├── test2015
            │       └── train2015
            ├── openimages
            │   └── images
            ├── pic
            │   └── image
            │       ├── train
            │       └── val
            └── vcoco
                ├── train2014
                └── val2014
    ```

2. Download the Bongard-HOI annotations from [here](http://tensorlab.cms.caltech.edu/projects/bongard-hoi/data/bongard_hoi_annotations.tar) and extract them to `./cache`

3. Download the detected bounding boxes from [here](http://tensorlab.cms.caltech.edu/projects/bongard-hoi/data/hico_faster_rcnn_R_101_DC5_3x_objectness.pkl) and extract them to `./cache`

4. Download the pretrained ResNet-50 from [here](http://tensorlab.cms.caltech.edu/projects/bongard-hoi/data/resnet.tar) and extract them to `./cache`
