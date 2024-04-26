# STDM-Net
### STDM-transformer: Space-time dual multi-scale transformer network for skeleton-based action recognition

# Data Preparation

 - NTU-60
    - Download the NTU-60 data from the https://github.com/shahroudy/NTURGB-D
    - Generate the train/test splits with `python prepare/ntu_60/gendata.py`
 - NTU-120
    - Download the NTU-120 data from the https://github.com/shahroudy/NTURGB-D
    - Generate the train/test splits with `python prepare/ntu_120/gendata.py`
 - Note
    - You can check the raw/generated skeletons through the function `view_raw/generated_skeletons_and_images()` for NTU and function `ske_vis()` for dhg/shrec in gendata.py


# Training & Testing

Change the config file depending on what you want.

    `python train_val_test/train.py --config ./config/shrec/shrec_dstanet_14.yaml`


Then combine the generated scores with: 

    `python train_val_test/ensemble.py`



## Dependencies

All the necessary dependencies for this project are specified in the `requirements.txt` file.