python probe.py \
  contact_map \
  --batch_size 2 \
	--learning_rate .00005 \
  --warmup_steps 2000 \
  --num_train_epochs 50 \
  --save_freq improvement \
  --patience 3 \
  --num_workers 0