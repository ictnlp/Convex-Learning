exp=vanilla_convex
data_dir=/path/to/binarized_dataset
checkpoint_dir=./checkpoints/$exp
plugin_path=/path/to/convex_learning_plugin
mle_pretrain_model=./checkpoints/vanilla
length_factor=0.01
convex_order=3

nohup fairseq-train $data_dir \
    --user-dir ${plugin_path} \
    --fp16 --ddp-backend=legacy_ddp \
    --finetune-from-model ${mle_pretrain_model} \
    --task translation_lev \
    --criterion nat_loss_convex --convex-order ${convex_order} \
    --arch nat_convex --left-pad-source --src-embedding-copy \
    --noise full_mask --share-all-embeddings \
    --decoder-learned-pos --encoder-learned-pos \
    --pred-length-offset --length-loss-factor ${length_factor} \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0003 --lr-scheduler inverse_sqrt \
    --stop-min-lr '1e-09' --warmup-updates 500 \
    --warmup-init-lr '1e-07' --label-smoothing 0.0 \
    --dropout 0.3 --weight-decay 0.01 \
    --max-tokens 8192 --update-freq 8 --max-update 10000 \
    --seed 0 \
    --valid-subset valid \
    --skip-invalid-size-inputs-valid-test \
    --fixed-validation-seed 7 \
    --keep-best-checkpoints 5 \
    --keep-interval-updates 5 --keep-last-epochs 5 \
    --save-dir ${checkpoint_dir} \
    --save-interval-updates 500 \
    --log-format 'simple' --log-interval 10 > logs/$exp.txt &
