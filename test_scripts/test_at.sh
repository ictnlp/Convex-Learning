data_dir=wmt14_ende_bin
checkpoint_path=checkpoints/at_convex
plugin_path=Convex-Learning

fairseq-generate ${data_dir} \
  --user-dir ${plugin_path} \
  --fp16 \
  --gen-subset test \
  --path ${checkpoint_path}/checkpoint_best.pt \
  --max-tokens 8192 --remove-bpe \
  --beam 5 --lenpen 0 --nbest 1 \
  --left-pad-source > out

grep ^H out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > ${checkpoint_path}/pred.de
perl test_scripts/multi-bleu.perl test_scripts/ref.de < ${checkpoint_path}/pred.de