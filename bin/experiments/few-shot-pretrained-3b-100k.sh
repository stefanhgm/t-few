allow_skip_exp=True
train_batch_size=8

# TODO: Set per experiment
cuda_device=1

# Set adaptively
num_steps=0 #  #  2000, 256 / 1k: 5000 4k (30 epochs)/16k (8 epochs): 16000 all (5 epochs): 60000 eol 320000 loh/surgery | balanced: 4k shot eol, 20000 loh, 135000 surgery
eval_epoch_interval=0

# TODO: Set per experiment
# Excluded 800000 for now since one epoch takes ~20h
# IBC  32 64 128 256 1024 4096 16348 50000 100000 800000
# External 32 64 128 256 512 1024 2048 4096 16348 50000
for num_shot in 32 64 128 256 512 1024 2048 4096
do
  # Determine epoch number xxx * for each shot size
  if [ "$num_shot" -le 64 ] ; then
    num_steps=$(( 250 * $num_shot / $train_batch_size))
  elif [ "$num_shot" -le 256 ] ; then
    num_steps=$(( 200 * $num_shot / $train_batch_size))
  elif [ "$num_shot" -le 2048 ] ; then
    num_steps=$(( 50 * $num_shot / $train_batch_size))
  elif [ "$num_shot" -le 16384 ] ; then
    num_steps=$(( 20 * $num_shot / $train_batch_size))
  else
    num_steps=$(( 8 * $num_shot / $train_batch_size))
  fi

  # Determine epoch eval number for each shot size
  if [ "$num_shot" -ge 2048 ] ; then
    eval_epoch_interval=1
  elif [ "$num_shot" -ge 64 ] ; then
    eval_epoch_interval=$(( 2048 / $num_shot ))
  else
    eval_epoch_interval=50
  fi

# TODO: Set per experiment
  # income car
  # eol_important_v_c_10 eol_important_v_c_999 eol_important_v_c_10_balanced eol_important_v_c_999_balanced loh_important_v_c_10 loh_important_v_c_999 loh_important_v_c_10_balanced loh_important_v_c_999_balanced surgery_important_v_c_10 surgery_important_v_c_999 surgery_important_v_c_10_balanced surgery_important_v_c_999_balanced
  # 2048: titanic titanic_list heart heart_list diabetes diabetes_list voting voting_list
  # 4096: wine wine_list car car_list
  # 50000: income income_list
  for dataset in wine wine_list car car_list
  do
    for seed in 42  # 1024 0 1 32
    do
      CUDA_VISIBLE_DEVICES=${cuda_device} CONFIG_PATH=/data/IBC/stefan_ibc/t-few/configs HF_HOME=/home/stefanhg/.cache/huggingface \
      python -m src.pl_train -c t03b.json+ia3.json+global.json -k dataset=${dataset} load_weight="pretrained_checkpoints/t03b_ia3_finish.pt" num_steps=${num_steps} num_shot=${num_shot} \
      exp_name=t03b_${dataset}_numshot${num_shot}_seed${seed}_ia3_pretrained100k few_shot_random_seed=${seed} seed=${seed} allow_skip_exp=${allow_skip_exp} eval_epoch_interval=${eval_epoch_interval}
    done
  done
done