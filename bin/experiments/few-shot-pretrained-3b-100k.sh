allow_skip_exp=True
train_batch_size=4
grad_accum_factor=$(( 8 / $train_batch_size))
lr=0.003  # Original in config 3e-3, also used as default in ia3

# TODO: Set per experiment
cuda_device=0
model='t011b'

# Set adaptively
num_steps=0 #  #  2000, 256 / 1k: 5000 4k (30 epochs)/16k (8 epochs): 16000 all (5 epochs): 60000 eol 320000 loh/surgery | balanced: 4k shot eol, 20000 loh, 135000 surgery
eval_epoch_interval=0

# TODO: Set per experiment
# Excluded 800000 for now since one epoch takes ~20h
# IBC  4 8 16 32 64 128 256 512 1024 2048 4096 16348 50000 100000 800000
# External 32 64 128 256 512 1024 2048 4096 16348 50000
for num_shot in 4 8 16 32 64 128 256 512 1024 2048 # 512  # 64 128 1024 50000  # 4 8 16 32 64 128 256 512 1024 2048 4096 16348 50000 100000
do

  num_steps=$(( 40 * $num_shot / $train_batch_size))
  eval_epoch_interval=1

  # Determine epoch number xxx * for each shot size
  # if [ "$num_shot" -le 64 ] ; then
  #   num_steps=$(( 80 * $num_shot / $train_batch_size))  # Formerly 250, 200, 50
  # elif [ "$num_shot" -le 256 ] ; then
  #   num_steps=$(( 60 * $num_shot / $train_batch_size))
  # elif [ "$num_shot" -le 2048 ] ; then
  #   num_steps=$(( 50 * $num_shot / $train_batch_size))
  # elif [ "$num_shot" -le 16384 ] ; then
  #   num_steps=$(( 20 * $num_shot / $train_batch_size))
  # else
  #   num_steps=$(( 8 * $num_shot / $train_batch_size))
  # fi

  # # Determine epoch eval number for each shot size
  # if [ "$num_shot" -ge 1024 ] ; then
  #   eval_epoch_interval=1
  # elif [ "$num_shot" -ge 64 ] ; then
  #   eval_epoch_interval=$(( 512 / $num_shot ))  # This are 8 for 64
  # else
  #   eval_epoch_interval=10  # For even smaller num shots than 64
  # fi

# TODO: Set per experiment
  # income car
  # eol_important_v_c_10 eol_important_v_c_999 eol_important_v_c_10_balanced eol_important_v_c_999_balanced loh_important_v_c_10 loh_important_v_c_999 loh_important_v_c_10_balanced loh_important_v_c_999_balanced surgery_important_v_c_10 surgery_important_v_c_999 surgery_important_v_c_10_balanced surgery_important_v_c_999_balanced
  # 2048: titanic titanic_list heart heart_list diabetes diabetes_list voting voting_list
  # 4096: wine wine_list car car_list
  # 50000: income income_list
  for dataset in income income_list # diabetes diabetes_list # income income_list #  eol_important_v_c_p_999 eol_list_important_v_c_p_999 loh_important_v_c_p_10 surgery_important_v_c_p_10
  do
    for seed in 42 1024 0 1 32
    do
      CUDA_VISIBLE_DEVICES=${cuda_device} CONFIG_PATH=/data/IBC/stefan_ibc/t-few/configs HF_HOME=/home/stefanhg/.cache/huggingface \
      python -m src.pl_train -c ${model}.json+ia3.json+global.json -k dataset=${dataset} load_weight="pretrained_checkpoints/${model}_ia3_finish.pt" num_steps=${num_steps} num_shot=${num_shot} \
      exp_name=${model}_${dataset}_numshot${num_shot}_seed${seed}_ia3_pretrained100k few_shot_random_seed=${seed} seed=${seed} allow_skip_exp=${allow_skip_exp} eval_epoch_interval=${eval_epoch_interval} \
      batch_size=${train_batch_size} grad_accum_factor=${grad_accum_factor} lr=${lr}
    done
  done
done
