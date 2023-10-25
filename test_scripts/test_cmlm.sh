data_dir=wmt14_ende_bin
checkpoint_path=checkpoints/cmlm_convex
plugin_path=Convex-Learning


fairseq-generate ${data_dir} \
    --user-dir ${plugin_path} \
    --gen-subset test \
    --task translation_lev \
    --path ${checkpoint_path}/checkpoint_best.pt \
    --iter-decode-with-beam 5 \
    --iter-decode-max-iter 0 \
    --iter-decode-eos-penalty 0 \
    --max-tokens 4096 --remove-bpe \
    --left-pad-source > out

grep ^H out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > ${checkpoint_path}/pred.de
perl test_scripts/multi-bleu.perl test_scripts/ref.de < ${checkpoint_path}/pred.de

