***Stage 2: Single-view reconstruction training***

In this stage, we first pre-train our model with synthetic images (ShapeNet renderings). Then the model is fine-tuned with real images.

**Dataset**

* ShapeNet Renderings. We render training data ourselves, adding lighting and pose variations. Please download the synthetic images of 13 categories in the link: https://drive.google.com/drive/folders/1fO8bh1v-HGTmMp9CC8l73bl2zPgRJG7A?usp=sharing.  The rendering script is adopted from the paper [DISN: Deep Implicit Surface Network for High-quality Single-view 3D Reconstruction](https://github.com/laughtervv/DISN). Please also cite it if you use the image data.

```bash
  Data structure:
     -- RGB Image      
     -- Albedo Image       
     -- Mask Image
     -- Camera Projection Matrix
```

**Pretrained models**

The pretrained models can be downloaded  from: https://drive.google.com/file/d/1Gjfcgq9IK-vkNEFdykhee1atdOatUk4m/view?usp=sharing

**Training**

