python ./train/train_qa.py \
  --context_file $1 \
  --train_file $2 \
  --validation_file $3 \
  --model_type bert \
  --model_name_or_path hfl/chinese-lert-base \
  --max_seq_length 512 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --num_train_epochs 11 \
  --output_dir model/qa_lert_base \
  --checkpointing_steps 'epoch' \
  --resume_from_checkpoint model/qa_lert_base/epoch_9 \
  --with_tracking