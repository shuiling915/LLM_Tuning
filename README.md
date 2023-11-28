# LLM-Tuning

一种平价的大模型实现方案，基于chatglm/llama/qwen等大模型 + LoRA 进行finetune.

数据集: [alpaca](https://github.com/tatsu-lab/stanford_alpaca)


## S1 Finetune

### 准备

- 显卡: 显存 >= 16G (最好24G或者以上)
- 环境：
- - python>=3.8
- - cuda>=11.6, cupti, cuDNN, TensorRT等深度学习环境
- - pip3 install -r requirements_LLM.txt -i https://pypi.douban.com/simple/


### 数据预处理

tokenization

```bash
python tokenize_dataset_rows.py \
    --json_path data/alpaca_data.json \
    --save_path data/alpaca \
    --model_path skyline2006/llama-7b/ \
    --version v1 \
    --num_examples 1500 \
    --max_seq_length 200 \ 
    --skip_overlength  False             
    
```

- `--json_path` 微调的数据路径, 格式json, 对每行的['context']和['target']字段进行encode
- `--save_path` 输出路径
- `--model_path` 导入模型的路径
- `--version` 模型的版本
- `--num_examples` 微调的样本数量
- `--max_seq_length` 样本的最大长度


### 训练

```bash
python finetune.py \
    --data_path data/alpaca \
    --output_dir output \
    --model_path model_path \
    --lora_rank 8 \
    --per_device_train_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --max_steps 52000 \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 50 \

```

### 推理

参考 [infer.ipynb](infer.ipynb)

Finetune前后对比

## S2. Reward Model

## S3. PPO


# TODO:
开发可视化界面
