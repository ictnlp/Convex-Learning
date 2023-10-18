data_dir=/path/to/binarized_dataset
average_checkpoint_path=/path/to/checkpoint
plugin_path=/path/to/convex_learning_plugin

fairseq-generate ${data_dir} \
  --user-dir ${plugin_path} \
  --fp16 \
  --gen-subset test \
  --path ${average_checkpoint_path} \
  --max-tokens 8192 --remove-bpe \
  --beam 5 --lenpen 0 --nbest 1 \
  --left-pad-source 