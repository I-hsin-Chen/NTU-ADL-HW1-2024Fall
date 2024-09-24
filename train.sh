python ./train/train_mc.py \
  --context_file $1 \
  --train_file $2 \
  --validation_file $3 \
  --model_name_or_path hfl/chinese-lert-base \
  --max_seq_length 512 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 3 \
  --learning_rate 3e-5 \
  --output_dir ./model/mc_lert_base \
  --checkpointing_steps 'epoch' \
  --with_tracking

python ./train/train_qa.py \
  --context_file $1 \
  --train_file $2 \
  --validation_file $3 \
  --model_name_or_path hfl/chinese-lert-base \
  --max_seq_length 512 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --num_train_epochs 10 \
  --output_dir model/qa_lert_base \
  --checkpointing_steps 'epoch' \
  --with_tracking
  # --resume_from_checkpoint model/qa_lert_base/epoch_9 \