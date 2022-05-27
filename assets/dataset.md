Data Preparation
===

1. Download the images from [HAKE](http://hake-mvig.cn/) dataset. You may follow the [official instruction](https://github.com/DirtyHarryLYL/HAKE/tree/master/Images#download-images-for-hake). For your convenience, you may download all the images required by Bongard-HOI [here](https://drive.google.com/file/d/1aqcp3XKB2KhyuS4rP91x_Yp6bZ_ELAk0/view?usp=sharing). The images should be extracted to `./assets/data/hake/images` and the file structure looks like:
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

2. Download the Bongard-HOI annotations from [here](https://drive.google.com/file/d/1AbMtJ7Wd6qlOT4qP2lIK9NQrKxCerKnR/view?usp=sharing) and extract them to `./cache`

3. Download the detected bounding boxes from [here](https://drive.google.com/file/d/1oJIFnhVJRmFKqSkFjmBLEU6jihurJldG/view?usp=sharing) and extract them to `./cache`

4. Download the pretrained ResNet-50 from [here](https://drive.google.com/file/d/18i0yWls81cPa_KNjSBUw08lbmDT29vo5/view?usp=sharing) and extract them ito `./cache`
