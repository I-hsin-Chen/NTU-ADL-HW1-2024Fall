# python ./inference_mc.py \
#     --model_name_or_path model/mc_bert_base/ \
#     --tokenizer_name model/mc_bert_base/ \
#     --test_file $2 \
#     --context_file $1 \
#     --output_dir output/ \
#     --max_seq_length 512 \

python ./inference_qa.py \
    --model_name_or_path model/qa_bert_base/ \
    --tokenizer_name model/qa_bert_base/ \
    --prediction_path $3 \
    --test_file $2 \
    --context_file $1 \
    --max_seq_length 512 \