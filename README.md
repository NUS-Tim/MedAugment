# MedAugment
<a href='https://arxiv.org/abs/2306.17466'><img src='https://img.shields.io/badge/ArXiv-2306.17466-red' /></a> 

Official Pytorch Implementation for Paper “MedAugment: Universal Automatic Data Augmentation Plug-in for Medical Image Analysis”

To use MedAugment as a plug-in for your own project, you should have a "baseline" folder for your custom dataset at:
```
./datasets/classification/your_dataset_name/baseline
```

The organization of the "baseline" folder should follow:
```
├── classification
    ├── your_dataset_name
        ├── baseline
            ├── training
            |   ├── class_1
            |   |   ├── img_1.jpg  # .png
            |   │   ├── img_2.jpg
            |   │   ├── ...
            |   ├── class_2
            |   |   ├── img_a.jpg
            |   │   ├── img_b.jpg
            |   │   ├── ...
            |   ├── ...
            └── validation
            └── test

├── segmentation
    ├── your_dataset_name
        ├── baseline
            ├── training
            |   ├── img_1.jpg
            │   ├── img_2.jpg
            │   ├── ...
            ├── training_mask
            |   ├── img_1_mask.jpg  # suffix
            │   ├── img_2_mask.jpg
            │   ├── ...
            └── validation
            └── validation_mask
            └── test
            └── test_mask
```

You can then move the [plug-in](./utils/medaugment.py) to your project utils and run:
```
python ./utils/medaugment.py --dataset=your_dataset_name
```
This will produce an augmented dataset named "medaugment" at the same level as the "baseline" folder 

To run MedAugment and other augmentation methods and train the model in the paper:
```
python ./utils/generation.py
python classification.py
```
You should have a "recording" folder at the root with two subfolders named "classification" and "segmentation"

##
If you find MedAugment useful for your research, please cite our paper as:
```
@misc{liu2023medaugment,
      title={MedAugment: Universal Automatic Data Augmentation Plug-in for Medical Image Analysis}, 
      author={Zhaoshan Liu and Qiujie Lv and Yifan Li and Ziduo Yang and Lei Shen},
      year={2023},
      eprint={2306.17466},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```
