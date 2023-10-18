data_dir=/path/to/binarized_dataset
average_checkpoint_path=/path/to/checkpoint
plugin_path=/path/to/convex_learning_plugin


python ${plugin_path}/scripts/generate_ctc.py ${data_dir} \
    --user-dir ${plugin_path} \
    --gen-subset test \
    --src-upsample-ratio 2 \
    --model-overrides "{\"src_upsample_ratio\":2,\"src_embedding_copy\":True,\"plain_ctc\":True}" \
    --task translation_ctc \
    --path ${average_checkpoint_path} \
    --iter-decode-max-iter 0 \
    --iter-decode-eos-penalty 0 \
    --max-tokens 2048 --remove-bpe \
    --left-pad-source