# Simple Efficient Differential Batch Augmentations for Classification and Object Detection (PyTorch)

Differential Augmentations can be used to improve detection and classification performance, generate robust samples in GAN setup [DiffAugment](\https://arxiv.org/abs/2006.10738)

#### The implementation is compatbile with
* [https://github.com/jwyang/faster-rcnn.pytorch]

### Augmentations Added
- ✅ Image aware Random Cutout (paper link)
- ✅ Jitter
- ✅ Flip
- ✅ Gaussian Noise

### Augmentations To-be Added
- ⬜️ Object Aware Random cutout (paper link)
- ⬜️ Others
- ⬜️ Compatibility with Classification 

### Use Augmentation Pipeline

```
from .augmentation_pipeline import AvengersAssemble
augmentations = AvengersAssemble()
org_data = [im_info, gt_boxes, num_boxes]
aug_img, aug_data = self.augmentations(org_img, org_data)
```
