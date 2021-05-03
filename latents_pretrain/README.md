***Stage 1: Shape and albedo decoders pre-training***

In this stage, the shape and albedo decoders are first pre-trained with CAD models from ShapeNet dataset.

--------------------------------------

**Dataset**

The decoders take point-value pairs as inputs. Please download the data from: https://drive.google.com/file/d/1Bnqneg6y7YdYnuzXYA9TCCvZQdJvRSjw/view?usp=sharing. Our sampling method used this paper [Learning Implicit Fields for Generative Shape Modeling](https://github.com/czq142857/implicit-decoder). Please consider citing it if your used the data. 

**Pretrained models**

Please download the pretrained models from: https://drive.google.com/file/d/1Gjfcgq9IK-vkNEFdykhee1atdOatUk4m/view?usp=sharing

**Training**

```bash
## The decoders are trained on gradually increasing resolution data ##
python main.py --train --epoch 50 --learning_rate 0.0005  --sample_reso 16
python main.py --train --epoch 50 --learning_rate 0.0001  --sample_reso 32
python main.py --train --epoch 50 --learning_rate 0.0001  --sample_reso 64
python main.py --train --epoch 50 --learning_rate 0.00001 --sample_reso 64
python main.py --get_latent
```

