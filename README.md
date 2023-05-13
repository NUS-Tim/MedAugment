# MedAugment
Official Pytorch Implementation for Paper: “MedAugment: Universal Automatic Data Augmentation Plug-in for Medical Image Analysis”

To use MedAugment as a plug-in for your own project, move the [plug-in](./utils/medaugment.py) to your project utils and run:
```
python ./utils/medaugment.py --dataset=your_dataset_name
```

This will produce an augmented dataset named "medaugment" at the same level as the "baseline" folder located at:
```
./datasets/classification/your_dataset_name/baseline
```

For classification, provided "baseline" folder should have three subfolders for training, validation, and test. 
For segmentation, provided baseline folder should have six subfolders, where three have the "_mask" suffix. 
For each subfolder, the image and corresponding mask (if exist) should be processed to have the identical resolution, that is, 224 × 224. 
The mask should be named with the suffix "_mask" following the image name. The accepted format includes PNG as well as JPG.

To run MedAugment and other augmentation methods and train the classification model in the paper:
```
python ./utils/generation.py
python classification.py
```
You should have a "recording" folder at the root with two subfolders named "classification" and "segmentation".

Important: Before running the commands, double-check the argument settings.