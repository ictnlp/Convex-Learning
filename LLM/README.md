# Convex Learning for LLMs

This repository contains the guideline of reproducing the results of [convex learning](https://openreview.net/forum?id=sla7V80uWA) on large language models (LLMs).

<details>
  <summary>Click Here for Performance on large language models</summary>

  ![open_generation_result](../images/llm_results.png)
</details>


## Requirements

This repository is based on open-sourced LLMs with HuggingFace's transformers library.

Framework Versions:
- Python 3.8.12
- Pytorch 1.13.1+cu117
- Transformers (git+https://github.com/huggingface/transformers.git) 
- Peft (git+https://github.com/huggingface/peft.git)
- Other requirements
```
pip install -r requirements.txt
```

## Preparation

The pre-trained large language models can be downloaded at HuggingFace model hub: 
[[LLaMA-7b]](https://huggingface.co/decapoda-research/llama-7b-hf)
[[LLaMA-13b]](https://huggingface.co/decapoda-research/llama-13b-hf). The instruction-following dataset [Alpaca](./train/data_alpaca_gpt4_hf_en.json) is already provided in this repository.

## Training

In the training scripts [train_mle.sh](./train_mle.sh) and [train_convex.sh](./train_convex.sh), specify `premodel=/path/to/llama` as the path to the pre-trained model. The following command utilizes LLaMA-7B as the foundation model and conduct instruction tuning with MLE loss:

```bash
# instruction tuning with MLE loss
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh train_mle.sh
```

The following command utilize LLaMA-7B as the foundation model and conduct instruction tuning with convex-composition loss, implemented in [run_convex.py](./train/run_convex.py#L79-L103).

```bash
# instruction tuning with convex-composition loss
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 sh train_convex.sh
```

## Decoding & Evaluation
After instruction tuning, the checkpoints are saved in [checkpoints](./checkpoints). The following test scripts evaluate the generative capability of LLMs on WMT22 machine translation benchmarks using the prompt `Translate the following sentences from [SRC] to [TGT]`, and save the sacrebleu scores in [results](./results).
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 sh test_mle.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 sh test_convex.sh
```
