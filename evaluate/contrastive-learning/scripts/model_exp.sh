model_names="kf-deberta-base roberta-base"
batch_sizes="bsz-4"

for MODEL_NAME in $model_names; do
    for BATCH_SIZE in $batch_sizes; do
        python main.py \
            default_config_path=configs/default.yaml \
            train_config_path=configs/train/${BATCH_SIZE}.yaml \
            model_config_path=configs/model/${MODEL_NAME}.yaml \
            run_name=${MODEL_NAME}-${BATCH_SIZE}
    done
done
