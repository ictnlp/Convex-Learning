data_dir=wmt14_ende_bin
exp=vanilla
checkpoint_dir=checkpoints/$exp
plugin_path=Convex-Learning

nohup fairseq-train $data_dir \
    --user-dir ${plugin_path} \
    --fp16 \
    --save-dir ${checkpoint_dir} \
    --ddp-backend=legacy_ddp \
    --task translation_lev \
    --criterion nat_loss \
    --arch nonautoregressive_transformer --left-pad-source --src-embedding-copy \
    --noise full_mask \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --stop-min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --pred-length-offset \
    --length-loss-factor 0.1 \
    --apply-bert-init \
    --log-format 'simple' --log-interval 100 \
    --fixed-validation-seed 7 \
    --max-tokens 8192 \
    --update-freq 2 \
    --save-interval-updates 10000 \
    --keep-interval-updates 5 --keep-last-epochs 5 \
    --max-update 300000 > logs/$exp.txt &