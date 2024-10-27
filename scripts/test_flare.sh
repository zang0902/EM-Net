CUDA_VISIBLE_DEVICES=3 python test.py  \
    --feature_size 32 --infer_overlap 0.3 \
    --pretrained_dir path_to_checkpoint \
    --pretrained_model_name model.pt \
    --data_config flare \
    --model_config em_net_c \
    --trainer_config test \
