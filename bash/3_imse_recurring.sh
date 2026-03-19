exp_name='recurring'
CUDA_VISIBLE_DEVICES=0 python main.py --cfg ./cfgs/vit/imse.yaml --exp_name $exp_name \
                                        RECURRING 10 RECURRING_TYPE split OPTIM.LR 3e-3