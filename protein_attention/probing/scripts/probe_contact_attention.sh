python probe.py \
  contact_map \
  --attention_probe \
  --batch_size 16 \
	--learning_rate .0001 \
  --warmup_steps 2000 \
  --num_train_epochs 50 \
  --save_freq improvement \
  --patience 3 \
  --num_workers 0 \
  --max_seq_len 512