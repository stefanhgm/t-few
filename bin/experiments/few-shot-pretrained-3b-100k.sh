num_steps=15000
allow_skip_exp=True
for num_shot in 16384 4096 1024 256
do
    for dataset in eol_important_c_v_10 eol_important_c_v_999 loh_important_c_v_10 loh_important_c_v_999 surgery_important_c_v_10 surgery_important_c_v_20 income car
    do
        for seed in 42 1024 0 1 32
        do
            CUDA_VISIBLE_DEVICES=0 CONFIG_PATH=/data/IBC/stefan_ibc/t-few/configs HF_HOME=/home/stefanhg/.cache/huggingface python -m src.pl_train -c t03b.json+ia3.json+global.json -k dataset=${dataset} load_weight="pretrained_checkpoints/t03b_ia3_finish.pt" num_steps=${num_steps} num_shot=${num_shot} exp_name=t03b_${dataset}_numshot${num_shot}_seed${seed}_ia3_pretrained100k few_shot_random_seed=${seed} seed=${seed} allow_skip_exp=${allow_skip_exp}
        done
    done
done
