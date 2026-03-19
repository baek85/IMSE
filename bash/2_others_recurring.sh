exp_name='recurring'
CUDA_VISIBLE_DEVICES=0 python main.py --cfg ./cfgs/vit/cotta.yaml --exp_name $exp_name \
                            RECURRING 10 RECURRING_TYPE split TEST.BATCH_SIZE 64 OPTIM.LR 1e-5
CUDA_VISIBLE_DEVICES=0 python main.py --cfg ./cfgs/vit/vida.yaml --checkpoint ./checkpoint/imagenet_vit_vida.pt --exp_name $exp_name \
                        RECURRING 10 RECURRING_TYPE split OPTIM.LR 6.25e-8 OPTIM.ViDALR 2e-7
CUDA_VISIBLE_DEVICES=0 python main.py --cfg ./cfgs/vit/tent.yaml --exp_name $exp_name \
                        RECURRING 10 RECURRING_TYPE split OPTIM.LR 3e-4
