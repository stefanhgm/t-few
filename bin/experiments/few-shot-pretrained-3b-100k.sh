#!/bin/bash
allow_skip_exp=True
# Public 4, IBC 1
train_batch_size=4
grad_accum_factor=$(( 8 / $train_batch_size))
lr=0.003  # Original in config 3e-3, also used as default in ia3
re='^[0-9]+$'

# TODO: Set per experiment
cuda_device=2

# Set adaptively
num_steps=0 #  #  2000, 256 / 1k: 5000 4k (30 epochs)/16k (8 epochs): 16000 all (5 epochs): 60000 eol 320000 loh/surgery | balanced: 4k shot eol, 20000 loh, 135000 surgery
eval_epoch_interval=0

# TODO: Set per experiment
# Excluded 800000 for now since one epoch takes ~20h
# IBC  4 8 16 32 64 128 256 512 1024 2048 4096 16348 50000 100000 800000
# External 32 64 128 256 512 1024 2048 4096 16348 50000
for model in 't011b' # 't03b'
do
  # ibc 16 64 256 1024 4096 16384
  # public 4 8 16 32 64 128 256 512
  for num_shot in 4 8 16 32 64 128 256 512 'all'
  do
    # Public
    # Simple setting to create epoch graphs for external datasets - 250 epochs
    if [[ $num_shot =~ $re ]]; then
      if [ "$num_shot" -le 4 ] ; then
        num_steps=$(( 10 * $num_shot))
      else
        num_steps=$(( 5 * $num_shot))
      fi
    fi

    # IBC
    # num_steps=$(( 10 * ($num_shot / $grad_accum_factor)))
    # num_steps=0

    eval_epoch_interval=10


    # TODO: Set per experiment
    # income car
    # eol_important_v_c_10 eol_important_v_c_999 eol_important_v_c_10_balanced eol_important_v_c_999_balanced loh_important_v_c_10 loh_important_v_c_999 loh_important_v_c_10_balanced loh_important_v_c_999_balanced surgery_important_v_c_10 surgery_important_v_c_999 surgery_important_v_c_10_balanced surgery_important_v_c_999_balanced
    # income income_list income_list_shuffled income_list_values
    # car car_list car_list_shuffled car_list_values
    # heart heart_list heart_list_shuffled heart_list_values
    # diabetes diabetes_list diabetes_list_shuffled diabetes_list_values
    # income_list_inflation_control income_list_inflation_features income_list_inflation_label income_list_inflation_both
    # eol_list_important_v_c_p_999 eol_important_v_c_p_999 eol_list_important_v_c_p_10 eol_list_permuted_important_v_c_p_999
    # loh_list_important_v_c_p_999 loh_important_v_c_p_999 loh_list_important_v_c_p_10 loh_list_permuted_important_v_c_p_999
    # surgery_list_important_v_c_p_999 surgery_important_v_c_p_999 surgery_list_important_v_c_p_10 surgery_list_permuted_important_v_c_p_999
    # eol_list_permuted_important_v_c_p_999 loh_list_permuted_important_v_c_p_999 surgery_list_permuted_important_v_c_p_999
    # diabetes diabetes_list diabetes_list_shuffled diabetes_list_values diabetes_list_permuted heart heart_list heart_list_shuffled heart_list_values heart_list_permuted car car_list car_list_shuffled car_list_values car_list_permuted income income_list income_list_shuffled income_list_values income_list_permuted
    # income_list income income_list_values income_list_shuffled income_list_permuted car_list_permuted car_list_values car_list_shuffled car_list car heart_list_permuted heart_list_values heart_list_shuffled heart_list heart diabetes_list_permuted diabetes_list_values diabetes_list_shuffled diabetes_list diabetes
    # car_list_permuted car_list_values car_list_shuffled heart_list_permuted heart_list_values heart_list_shuffled diabetes_list_permuted car heart car_list heart_list
    for dataset in heart_list_shuffled_values diabetes_list_shuffled_values car_list_shuffled_values income_list_shuffled_values
    do
      # For ALL run
      if ! [[ $num_shot =~ $re ]]; then
        if [[ $dataset = *"income"* ]]; then
          num_steps=198000
        fi
        if [[ $dataset = *"car"* ]]; then
          num_steps=7200
        fi
        if [[ $dataset = *"heart"* ]]; then
          num_steps=3900
        fi
        if [[ $dataset = *"diabetes"* ]]; then
          num_steps=3300
        fi
      fi

      for seed in 42 1024 0 1 32 # 45 655 186 126 836
      do
        CUDA_VISIBLE_DEVICES=${cuda_device} CONFIG_PATH=/localdata/stefanhg/t-few/configs HF_HOME=/home/stefanhg/.cache/huggingface \
        python -m src.pl_train -c ${model}.json+ia3.json+global.json -k dataset=${dataset} load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" num_steps=${num_steps} num_shot=${num_shot} \
        exp_name=${model}_${dataset}_numshot${num_shot}_seed${seed}_ia3_pretrained100k few_shot_random_seed=${seed} seed=${seed} allow_skip_exp=${allow_skip_exp} eval_epoch_interval=${eval_epoch_interval} \
        batch_size=${train_batch_size} grad_accum_factor=${grad_accum_factor} lr=${lr}
      done
    done
  done
done
