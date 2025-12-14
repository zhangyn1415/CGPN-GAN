# CLIP-Guided Generative Network for Pathology Nuclei Image Augmentation (CGPN-GAN)

# Descriptions

Nuclei segmentation and classification play a crucial role in the quantitative analysis of computational pathology (CPath). However, the challenge of creating a large volume of labeled pathology nuclei images due to annotation costs has significantly limited the performance of deep learning-based nuclei segmentation methods. Generative data augmentation offers a promising solution by substantially expanding the available training data without additional annotations. In medical image analysis, Generative Adversarial Networks (GANs) were effective for data augmentation, enhancing model performance by generating realistic synthetic data. However, these approaches lack scalability for multi-class data, as nuclei masks cannot provide sufficient information for diverse image generation. Recently, visual-language foundation models, pretrained on large-scale image-caption pairs, have demonstrated robust performance in pathological diagnostic tasks. In this study, we propose a CLIP-guided generative data augmentation method for nuclei segmentation and classification, leveraging the pretrained pathological CLIP text and image encoders in both the generator and discriminator. Specifically, we first create text descriptions by processing paired histopathology images and nuclei masks, which include information such as organ tissue type, cell count, and nuclei types. These paired text descriptions and nuclei masks are then fed into our multi-modal conditional image generator to guide the synthesis of realistic histopathology images. To ensure the quality of synthesized images, we utilize a high-resolution image discriminator and a CLIP image encoder-based discriminator, focusing on both local and global features of histopathology images. The synthetic histopathology images, paired with corresponding nuclei masks, are integrated into the real dataset to train the nuclei segmentation and classification model. Our experiments, conducted on diverse publicly available pathology nuclei datasets, including both qualitative and quantitative analysis, demonstrate the effectiveness of our proposed method. The experimental results of the nuclei segmentation and classification task underscore the advantages of our data augmentation approach.

### CLIP model
QuiltNet (https://huggingface.co/wisdomik/QuiltNet-B-32)

### Datasets
1. PanNuke (https://warwick.ac.uk/fac/sci/dcs/research/tia/data/pannuke)
2. Lizard (https://www.kaggle.com/datasets/aadimator/lizard-dataset)
3. EndoNuke (https://endonuke.ispras.ru/)
4. PUMA (https://zenodo.org/records/13859989)
5. NuInsSeg (https://www.kaggle.com/datasets/ipateam/nuinsseg)
6. MonuSeg (https://www.kaggle.com/datasets/tuanledinh/monuseg2018)

##----Quick Start----##

#### 1.Textual description generation
### For the PanNuke dataset, run the preprocessing script:
 preprocess.py 

## 2.Training
python  train.py

## 3.Sampling
python test.py

## 4.Downstream tasks
The synthetic images can be used to augment real datasets for training downstream models. Our implementation for nuclei segmentation is based on [HoVer‑Net]. (https://github.com/xinyiyu/Nudiff/tree/main).

---
### Citing CGPN-GAN

**Reference**

- [Diffusion-based Data Augmentation for Nuclei Image Segmentation] (https://github.com/xinyiyu/Nudiff)
- [Semantic Image Synthesis with Spatially-Adaptive Normalization] (https://github.com/NVlabs/SPADE)
- [GALIP: Generative Adversarial CLIPs for Text-to-Image Synthesis] (https://github.com/tobran/GALIP)


**Citation**

## If you find our work useful, please cite: 
CLIP-Guided Generative Network for Pathology Nuclei Image Augmentation, Medical Image Analysis (2025), 
Yanan Zhang, Qingyang Liu, Qian Chen, Xiangzhi Bai,
doi:https://doi.org/10.1016/j.media.2025.103908