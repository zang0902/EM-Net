CUDA_VISIBLE_DEVICES=0 python test.py  \
    --feature_size 32 --infer_overlap 0.3 \
    --pretrained_dir path_to_checkpoint \
    --pretrained_model_name model.pt \
    --data_config synapse \
    --model_config em_net \
    --trainer_config test 
