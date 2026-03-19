exp_name='ctta'

CUDA_VISIBLE_DEVICES=0 python main.py --cfg ./cfgs/vit/source.yaml --exp_name $exp_name
CUDA_VISIBLE_DEVICES=0 python main.py --cfg ./cfgs/vit/cotta.yaml --exp_name $exp_name
CUDA_VISIBLE_DEVICES=0 python main.py --cfg ./cfgs/vit/tent.yaml --exp_name $exp_name
CUDA_VISIBLE_DEVICES=0 python main.py --cfg ./cfgs/vit/vida.yaml --checkpoint ./checkpoint/imagenet_vit_vida.pt --exp_name $exp_name
