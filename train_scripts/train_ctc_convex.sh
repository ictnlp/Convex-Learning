data_dir=wmt14_ende_bin
exp=ctc_convex
checkpoint_dir=checkpoints/$exp
plugin_path=convex-learning
mle_pretrain_model=checkpoints/ctc/checkpoint_last.pt
convex_order=3

nohup fairseq-train $data_dir \
    --user-dir ${plugin_path} \
    --fp16 \
    --finetune-from-model ${mle_pretrain_model} \
    --save-dir ${checkpoint_dir} \
    --ddp-backend=legacy_ddp \
    --task translation_ctc \
    --criterion nat_loss_convex --left-pad-source \
    --src-embedding-copy \
    --src-upsample-ratio 2 --plain-ctc \
    --arch nat_ctc_convex --convex-order $convex_order \
    --noise full_mask \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0003 --lr-scheduler inverse_sqrt \
    --stop-min-lr '1e-09' --warmup-updates 500 \
    --warmup-init-lr '1e-07' --label-smoothing 0.0 \
    --dropout 0.1 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --log-format 'simple' --log-interval 10 \
    --fixed-validation-seed 7 \
    --max-tokens 4096 \
    --update-freq 16 \
    --save-interval-updates 500 \
    --keep-interval-updates 5 --keep-last-epochs 5 \
    --max-update 10000 > logs/$exp.txt &