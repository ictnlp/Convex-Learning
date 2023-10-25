exp=your_exp_name
data_dir=/path/to/binarized_dataset
checkpoint_dir=./checkpoints/$exp
plugin_path=/path/to/convex_learning_plugin

nohup fairseq-train $data_dir \
    --user-dir ${plugin_path} \
    --fp16 \
    --save-dir ${checkpoint_dir} \
    --task translation \
    --criterion label_smoothed_cross_entropy --left-pad-source \
    --arch transformer_wmt_en_de --share-all-embeddings \
    --convex-order ${convex_order} --label-smoothing 0.1 --dropout 0.1 \
    --optimizer adam  --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr '1e-07' --warmup-updates 4000 \
    --lr 0.0007 --stop-min-lr '1e-09' \
    --weight-decay 0.0 --max-tokens 8192  --update-freq 1 \
    --no-progress-bar --log-format json --log-interval 10 \
    --save-interval-updates 5000 --keep-interval-updates 5 --keep-last-epochs 5 \
    --max-update 150000 >> logs/$exp.txt & 