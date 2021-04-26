***Stage 1: Shape and albedo decoders pre-training***

In this stage, the shape and albedo decoders are first pre-trained with CAD models, which is vital for converging to faithful solutions.

--------------------------------------

**Dataset**



**Pretrained models**

P

**Training**

```bash
## The decoders are trained on gradually increasing resolution data ##
python main.py --train --epoch 50 --learning_rate 0.0005  --sample_reso 16
python main.py --train --epoch 50 --learning_rate 0.0001  --sample_reso 32
python main.py --train --epoch 50 --learning_rate 0.0001  --sample_reso 64
python main.py --train --epoch 50 --learning_rate 0.00001 --sample_reso 64
python main.py --get_latent
```

