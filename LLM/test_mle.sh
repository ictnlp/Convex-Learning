#!/bin/bash

model_name=llama_7b_mle

no_repeat_ngram_size=12

start_step=200
sep_step=200
end_step=2400

result_dir=results/${model_name}
log_name=log

if [ $no_repeat_ngram_size -gt 0 ]; then
  result_dir="${result_dir}_ngram${no_repeat_ngram_size}"
fi

if [ ! -d $result_dir ]; then
    mkdir -p $result_dir
    chmod 777 $result_dir -R
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRANSFORMERS_CACHE=train/cache/
export HF_HOME=train/cache/
export TF_ENABLE_ONEDNN_OPTS=0

inference()
{
    src=$1
    tgt=$2
    step=$3
    echo "src" $src
    echo "tgt" $tgt
    echo "step" $step
    log_name=${log_name}_step${step}
    python test/inference.py \
        --model-name-or-path checkpoints/$model_name/checkpoint-${step} \
        -lp $src-$tgt \
        -b 4 \
        -sa 'beam' \
        --batch 1 \
        --no-repeat-ngram-size $no_repeat_ngram_size \
        -ins test/instruct_inf.txt \
        -i test/WMT22/newstest22.$src-$tgt.$src \
        -tp "prompt_input" \
        -o $result_dir/${src}-${tgt}_step${step}.out 

    python test/sacre_verbose.py $result_dir/${src}-${tgt}_step${step}.out.hyp test/WMT22/newstest22.${src}-${tgt}.${tgt} $result_dir/tmp_bleu $tgt >> $result_dir/$log_name
}


for step in `seq $start_step $sep_step $end_step`; do

echo "step=" $step

(export CUDA_VISIBLE_DEVICES=0;inference de en $step;sleep 150)& \
(export CUDA_VISIBLE_DEVICES=1;inference en de $step)& \
(export CUDA_VISIBLE_DEVICES=2;inference en zh $step;sleep 150)& \
(export CUDA_VISIBLE_DEVICES=3;inference zh en $step)
wait
done
