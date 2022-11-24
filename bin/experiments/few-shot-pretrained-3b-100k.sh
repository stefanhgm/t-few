#!/bin/bash
allow_skip_exp=True
eval_before_training=True
balanced_ibc=True

# IBC 1
# train_batch_size=1
# grad_accum_factor=$(( 8 / $train_batch_size))

# Public 4
train_batch_size=4
grad_accum_factor=1

lr=0.003  # Original in config 3e-3, also used as default in ia3
re='^[0-9]+$'

# TODO: Set per experiment
cuda_device=0

# Set adaptively
num_steps=0 #  #  2000, 256 / 1k: 5000 4k (30 epochs)/16k (8 epochs): 16000 all (5 epochs): 60000 eol 320000 loh/surgery | balanced: 4k shot eol, 20000 loh, 135000 surgery
eval_epoch_interval=0

# TODO: Set per experiment
for model in 't011b' # 't03b'
do
  # ibc 16 64 256 1024 4096 16384
  # public 4 8 16 32 64 128 256 512
  # TODO: Set per experiment
  for num_shot in 4 8 16 32 64 128 256 512 # 64 256 1024 4096 16384 all
  do
    # income_list income income_list_values income_list_shuffled income_list_permuted car_list_permuted car_list_values car_list_shuffled car_list car heart_list_permuted heart_list_values heart_list_shuffled heart_list heart diabetes_list_permuted diabetes_list_values diabetes_list_shuffled diabetes_list diabetes
    # car_list_permuted car_list_values car_list_shuffled heart_list_permuted heart_list_values heart_list_shuffled diabetes_list_permuted car heart car_list heart_list
    # diabetes diabetes_list diabetes_list_shuffled diabetes_list_values diabetes_list_permuted heart heart_list heart_list_shuffled heart_list_values heart_list_permuted car car_list car_list_shuffled car_list_values car_list_permuted income income_list income_list_shuffled income_list_values income_list_permuted
    # eol_list_zero_shot_adaptive_256 eol_list_zero_shot_adaptive_4096 eol_list_zero_shot_age_sex_gender_race eol_list_zero_shot_least_frequent eol_list_zero_shot_most_frequent eol_list_zero_shot_most_frequent_conditions eol_list_zero_shot_most_frequent_procedures eol_list_zero_shot_oldest_concept eol_list_zero_shot_recent_concept
  # TODO: Set per experiment
    # for dataset in eol_list_zero_shot_age_sex_gender_race eol_list_zero_shot_least_frequent eol_list_zero_shot_least_frequent_conditions eol_list_zero_shot_least_frequent_procedures eol_list_zero_shot_most_frequent eol_list_zero_shot_most_frequent_conditions eol_list_zero_shot_most_frequent_procedures eol_list_zero_shot_oldest eol_list_zero_shot_oldest_conditions eol_list_zero_shot_oldest_procedures eol_list_zero_shot_recent eol_list_zero_shot_recent_conditions eol_list_zero_shot_recent_procedures eol_list_zero_shot_most_frequent_conditions_snomed eol_list_zero_shot_most_frequent_conditions_chv eol_list_zero_shot_most_frequent_conditions_icd eol_list_zero_shot_most_frequent_conditions_jargon eol_list_zero_shot_most_frequent_conditions_lay eol_list_zero_shot_most_frequent_conditions_medcin eol_list_zero_shot_most_frequent_conditions_shortened loh_list_zero_shot_age_sex_gender_race loh_list_zero_shot_least_frequent loh_list_zero_shot_least_frequent_conditions loh_list_zero_shot_least_frequent_procedures loh_list_zero_shot_most_frequent loh_list_zero_shot_most_frequent_conditions loh_list_zero_shot_most_frequent_procedures loh_list_zero_shot_oldest loh_list_zero_shot_oldest_conditions loh_list_zero_shot_oldest_procedures loh_list_zero_shot_recent loh_list_zero_shot_recent_conditions loh_list_zero_shot_recent_procedures loh_list_zero_shot_most_frequent_conditions_snomed loh_list_zero_shot_most_frequent_conditions_chv loh_list_zero_shot_most_frequent_conditions_icd loh_list_zero_shot_most_frequent_conditions_jargon loh_list_zero_shot_most_frequent_conditions_lay loh_list_zero_shot_most_frequent_conditions_medcin loh_list_zero_shot_most_frequent_conditions_shortened surgery_list_zero_shot_age_sex_gender_race surgery_list_zero_shot_least_frequent surgery_list_zero_shot_least_frequent_conditions surgery_list_zero_shot_least_frequent_procedures surgery_list_zero_shot_most_frequent surgery_list_zero_shot_most_frequent_conditions surgery_list_zero_shot_most_frequent_procedures surgery_list_zero_shot_oldest surgery_list_zero_shot_oldest_conditions surgery_list_zero_shot_oldest_procedures surgery_list_zero_shot_recent surgery_list_zero_shot_recent_conditions surgery_list_zero_shot_recent_procedures  surgery_list_zero_shot_most_frequent_conditions_snomed surgery_list_zero_shot_most_frequent_conditions_chv surgery_list_zero_shot_most_frequent_conditions_icd surgery_list_zero_shot_most_frequent_conditions_jargon surgery_list_zero_shot_most_frequent_conditions_lay surgery_list_zero_shot_most_frequent_conditions_medcin surgery_list_zero_shot_most_frequent_conditions_shortened
    # for dataset in income_gpt income_list_permuted income_list_shuffled car_gpt car_ttt car_t0 heart_list_permuted heart_list_shuffled heart_gpt heart_t0 heart_ttt diabetes_list_permuted diabetes_list_shuffled diabetes_gpt diabetes_t0 diabetes_ttt
    # for dataset in car_gpt car_t0 car_ttt car heart_gpt heart_t0 heart_ttt heart heart_list heart_list_permuted heart_list_shuffled diabetes_gpt diabetes_t0 diabetes_ttt diabetes diabetes_list diabetes_list_permuted diabetes_list_shuffled
    # for dataset in jungle jungle_list income heart diabetes creditg creditg_list car calhousing calhousing_list blood blood_list bank bank_list
    for dataset in jungle creditg calhousing blood bank
    do
      # IBC
      # num_steps=$(( 3 * ($num_shot / $grad_accum_factor)))
      # eval_epoch_interval=3
      # if [ "$num_shot" -le 512 ] ; then
      #   num_steps=$(( 10 * ($num_shot / $grad_accum_factor)))
      #   eval_epoch_interval=10
      # fi
      # eval_epoch_interval=999
      # eval_before_training=False
      # balanced_ibc=False
      # dataset="${dataset}_${num_shot}"
      # Zero shot
      # eval_before_training=True
      # num_steps=0

      # Public
      # Simple setting to run for fixed number of epochs
      # if [[ $num_shot =~ $re ]]; then
      #   if [ "$num_shot" -le 4 ] ; then
      #     num_steps=$(( 30 * ($num_shot / $train_batch_size)))
      #   else
      #     num_steps=$(( 5 * $num_shot))
      #   fi
      # fi

      # Zero-shot
      # eval_before_training=True
      # num_steps=0
      # Few-shot
      eval_before_training=False
      num_steps=$(( 30 * ($num_shot / $train_batch_size)))
      eval_epoch_interval=30

      #if [[ $dataset = *"adaptive"* ]]; then
      #  dataset="${dataset}_${num_shot}"
      #fi
      #if [[ $dataset = *"fixed"* ]]; then
      #  dataset="${dataset}_${num_shot}"
      #fi

      # For ALL run
      if ! [[ $num_shot =~ $re ]]; then
        if [[ $dataset = *"income"* ]]; then
          num_steps=295000
        fi
        if [[ $dataset = *"car"* ]]; then
          num_steps=10500
        fi
        if [[ $dataset = *"heart"* ]]; then
          num_steps=5600
        fi
        if [[ $dataset = *"diabetes"* ]]; then
          num_steps=4700
        fi
      fi

      for seed in 42 1024 0 1 32 # 45 655 186 126 836
      do
        CUDA_VISIBLE_DEVICES=${cuda_device} CONFIG_PATH=/root/t-few/configs HF_HOME=/root/.cache/huggingface \
        python -m src.pl_train -c ${model}.json+ia3.json+global.json -k dataset=${dataset} load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" num_steps=${num_steps} num_shot=${num_shot} \
        exp_name=${model}_${dataset}_numshot${num_shot}_seed${seed}_ia3_pretrained100k few_shot_random_seed=${seed} seed=${seed} allow_skip_exp=${allow_skip_exp} eval_before_training=${eval_before_training} eval_epoch_interval=${eval_epoch_interval} \
        batch_size=${train_batch_size} grad_accum_factor=${grad_accum_factor} lr=${lr}
      done
    done
  done
done
