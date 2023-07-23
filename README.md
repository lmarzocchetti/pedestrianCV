# pedestrianCV

Project for the Computer Vision course in the University of Modena and Reggio Emilia.

There are 3 parts:
  - Tracking
  - Action Recognition
  - Inpaint with Stable diffusion and ControlNet

### Requirements
There are some packages to preinstall to make this project working (see requirements.txt)

There are some system packages to install too:
```
conda install -c conda-forge gst-plugins-base
conda install -c nvidia libnpp
```

### Usage tracking and har
For the part of tracking and human action recognition:
```
python tracking.py -i [name_file_input] -o [name_file_output]
```
The name file must be passed without extensions and the file must be in the folder:
```
deep_sort/data/input_videos
```

### Usage stable diffusion inpaint
Go to the folder:
```
stable_diffusion
```
And the launch:
```
python inpaint.py 
```
Use the option ```--help``` to see the parameters that accept this script
