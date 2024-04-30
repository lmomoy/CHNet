# A Concise but High-performing Network for Image Guided Depth Completion in Autonomous Driving
This repository is the implementation of our paper [A Concise but High-performing Network for Image Guided Depth Completion in Autonomous Driving]().

## Demo
<p align="center">
  <img src="demo.gif" alt="example input output gif" width="1920" />
</p>

## Results
<p align="center">
  <img src="results.png" alt="example input output gif" width="500" />
</p>

## Dependent Environment
You can refer to the following environment:
+ python=3.6.2
+ torch==1.9.0+cu111
+ torchvision==0.10.0+cu111
```bash
pip install numpy matplotlib Pillow
pip install scikit-image
pip install opencv-contrib-python
```

## Data
- Download the [KITTI Depth](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) Dataset from their website. Use the following scripts to extract corresponding RGB images from the raw dataset. 
```bash
./download/rgb_train_downloader.sh
./download/rgb_val_downloader.sh
```
The downloaded rgb files will be stored in the `../data/data_rgb` folder. The overall code, data, and results directory is structured as follows.
```
├── CHNet
├── data
|   ├── data_depth_annotated
|   |   ├── train
|   |   ├── val
|   ├── data_depth_velodyne
|   |   ├── train
|   |   ├── val
|   ├── depth_selection
|   |   ├── test_depth_completion_anonymous
|   |   ├── test_depth_prediction_anonymous
|   |   ├── val_selection_cropped
|   └── data_rgb
|   |   ├── train
|   |   ├── val
├── results
```

## Train 
You can train the CHNet through the following command:
```
python main.py -b 8 (8 is a example of batch size)
```
## Evalution
You can evaluate the CHNet through the following command:
```
python main.py -b 1 -n e --evaluate [checkpoint-path]
```
## Test
You can test the CHNet through the following command for online submission:
```
python main.py -b 1 -n e --evaluate [checkpoint-path] --test
```

## Acknowledgement
Many thanks to these excellent opensource projects 
* [PENet](https://github.com/JUGGHM/PENet_ICRA2021)
* [GuideNet](https://github.com/kakaxi314/GuideNet)
* [self-supervised-depth-completion](https://github.com/fangchangma/self-supervised-depth-completion)
