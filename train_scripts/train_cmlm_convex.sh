exp=your_exp_name
data_dir=/path/to/binarized_dataset
checkpoint_dir=./checkpoints/$exp
plugin_path=/path/to/convex_learning_plugin
mle_pretrain_model=/path/to/pretrain_model
length_factor=0.01
convex_order=3

nohup fairseq-train $data_dir \
    --user-dir ${plugin_path} \
    --fp16 \
    --save-dir ${checkpoint_dir} \
    --finetune-from-model ${mle_pretrain_model} \
    --ddp-backend=legacy_ddp \
    --task translation_lev \
    --criterion nat_loss_convex --left-pad-source \
    --pred-length-offset --length-loss-factor 0.01 \
    --arch cmlm_transformer_convex --convex-order ${convex_order} \
    --noise random_mask --length-loss-factor ${length_factor} \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0003 --lr-scheduler inverse_sqrt \
    --stop-min-lr '1e-09' --warmup-updates 500 \
    --warmup-init-lr '1e-07' \
    --dropout 0.3 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --log-format 'simple' --log-interval 10 \
    --fixed-validation-seed 7 \
    --max-tokens 8192 \
    --update-freq 8 \
    --save-interval-updates 500 \
    --keep-interval-updates 5 --keep-last-epochs 5 \
    --max-update 10000 > logs/$exp.txt &