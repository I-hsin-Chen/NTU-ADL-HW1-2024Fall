python ./inference_mc.py \
    --model_name_or_path models/mc/ \
    --test_file $2 \
    --context_file $1 \
    --max_seq_length 512 \

python ./inference_qa.py \
    --model_name_or_path models/qa \
    --prediction_path $3 \
    --test_file $2 \
    --context_file $1 \
    --max_seq_length 512 \