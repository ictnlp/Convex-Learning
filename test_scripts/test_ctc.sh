data_dir=wmt14_ende_bin
checkpoint_path=checkpoints/ctc_convex
plugin_path=convex-learning


python ${plugin_path}/scripts/generate_ctc.py ${data_dir} \
    --user-dir ${plugin_path} \
    --gen-subset test \
    --src-upsample-ratio 2 \
    --model-overrides "{\"src_upsample_ratio\":2,\"src_embedding_copy\":True,\"plain_ctc\":True}" \
    --task translation_ctc \
    --path ${checkpoint_path}/checkpoint_best.pt \
    --iter-decode-max-iter 0 \
    --iter-decode-eos-penalty 0 \
    --max-tokens 2048 --remove-bpe \
    --left-pad-source > out

grep ^H out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > ${checkpoint_path}/pred.de
perl test_scripts/multi-bleu.perl test_scripts/ref.de < ${checkpoint_path}/pred.de