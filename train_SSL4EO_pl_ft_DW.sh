#!/usr/bin/env bash
python py_scripts_SSL4EO/train_SSL4EO_pl_ft_DWOSM.py \
        --dataset DW \
        --train_path $train_lmdb_path \
        --test_path $test_lmdb_path \
        --input_type s1/s2 \
        --label_type dw \
        --if_mask \
        --save_dir $save_dir \
        --model_framework deeplabv3plus \
        --model_type resnet50 \
        --n_channels 13 \
        --n_classes 9 \
        --loss_type cd \
        --n_train 0 \
        --validation 0 \
        --batch_size  32 \
        --test_batch_size 100 \
        --num_workers 8 \
        --accelerator gpu \
        --pretrain_tag mfs1/lfs1/mfs2/lfs2 \
        --weight_path $pretrain_weight_path \
        --weight_epoch 200 \
        --weight_load_tag encoder \
        --weight_batch_size 128 \
        --weight_lr_adjust_type rop \
        --weight_lr_adjust_value 30 \
        --encoder_finetune_tag efix/etr \
        --learning_rate 0.0005 \
        --scheduler_type cos \
        --scheduler_values 40 \
        --epochs 50 \
        --optimizer adam \
        --weight_label_smooth \
        --weight_label_smoothing_factor 0.15 0.05 \
        --weight_label_smoothing_prior_type ts \
        --weight_sample_selection \
        --weight_sample_selection_rmup_func exp \
        --weight_sample_selection_rmdown_epoch 80 \
        --weight_sample_selection_prop 0.5 \
        --weight_sample_selection_confidence_type ce 