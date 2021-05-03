### Fully Understanding Generic Objects: Modeling, Segmentation, and Reconstruction 

CVPR 2021. [[Arxiv](https://arxiv.org/abs/2104.00858), [PDF](http://cvlab.cse.msu.edu/pdfs/liu_tran_liu_cvpr2021.pdf), [Supp](http://cvlab.cse.msu.edu/pdfs/liu_tran_liu_cvpr2021_supp.pdf), [Project](http://cvlab.cse.msu.edu/project-fully3dobject.html)]

**[Feng Liu](https://zobject.org/),  [Luan Tran](http://www.cse.msu.edu/~tranluan/),  [Xiaoming Liu](http://cvlab.cse.msu.edu/pages/people.html)**

This Code is developed with Python3 and PyTorch 1.1

### Introduction

Inferring 3D structure of a generic object from a 2D image is a long-standing objective of computer vision. Conventional approaches either learn completely from CAD-generated synthetic data, which have difficulty in inference from real images, or generate 2.5D depth image via intrinsic decomposition, which is limited compared to the full 3D reconstruction. One fundamental challenge lies in how to leverage numerous real 2D images without any 3D ground truth. To address this issue, we take an alternative approach with semi-supervised learning. That is, for a 2D image of a generic object, we decompose it into latent representations of category, shape and albedo, lighting and camera projection matrix, decode the representations to segmented 3D shape and albedo respectively, and fuse these components to render an image well approximating the input image. Using a category-adaptive 3D joint occupancy field (JOF), we show that the complete shape and albedo modeling enables us to leverage real 2D images in both modeling and model fitting. The effectiveness of our approach is demonstrated through superior 3D reconstruction from a single image, being either synthetic or real, and shape segmentation.

### Citation

If you find our work useful in your research, please consider citing:

	@inproceedings{liu2021fully,
	  title={Fully Understanding Generic Objects: Modeling, Segmentation, and Reconstruction},
	  author={Liu, Feng and Tran, Luan and Liu, Xiaoming},
	  booktitle={Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
	  year={2021}
	}

### Training

***Stage 1: Shape and albedo decoders pre-training***

please refer to latents_pretrain/README

***Stage 2: Single-view reconstruction training***

please refer to SVR/README

### Contact

For questions feel free to post here or directly contact the author via liufeng6@msu.edu

