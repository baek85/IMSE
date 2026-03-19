exp_name='ctta'
CUDA_VISIBLE_DEVICES=0 python main.py --cfg ./cfgs/vit/imse.yaml --exp_name $exp_name ORDER default \
                                        SETTING continual SVD.LOAD_OPTIMIZER_STATE reset
                                        
CUDA_VISIBLE_DEVICES=0 python main.py --cfg ./cfgs/vit/imse.yaml --exp_name $exp_name ORDER default \
                                        SETTING continual SVD.DYNAMIC_MODE mix_adapt \
                                        SVD.LOAD_OPTIMIZER_STATE reset \
                                        SVD.TEMP 1.0