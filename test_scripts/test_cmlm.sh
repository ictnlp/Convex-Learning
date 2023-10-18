data_dir=/path/to/binarized_dataset
average_checkpoint_path=/path/to/checkpoint
plugin_path=/path/to/convex_learning_plugin


fairseq-generate ${data_dir} \
    --user-dir ${plugin_path} \
    --gen-subset test \
    --task translation_lev \
    --path ${average_checkpoint_path} \
    --iter-decode-with-beam 5 \
    --iter-decode-max-iter 0 \
    --iter-decode-eos-penalty 0 \
    --max-tokens 4096 --remove-bpe \
    --left-pad-source 