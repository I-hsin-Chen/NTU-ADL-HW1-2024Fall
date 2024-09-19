python ./inference_mc.py \
    --model_name_or_path model/multiple_choice/ \
    --tokenizer_name model/multiple_choice/ \
    --test_file $2 \
    --context_file $1 \
    --output_dir output/ \
    --max_seq_length 512 \