import argparse
import json
from tqdm import tqdm
from modelscope import AutoTokenizer, AutoConfig
import datasets


def format_example(example: dict) -> dict:
    context = f"Instruction: {example['instruction']}\n"
    if example.get("input"):
        context += f"Input: {example['input']}\n"
    context += "Answer: "
    target = example["output"]
    return {"context": context, "target": target}

def preprocess(tokenizer, config, example, max_seq_length, version):
    if version == 'v1':
        prompt = example["context"]
        target = example["target"]
        prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
        target_ids = tokenizer.encode(
            target,
            max_length=max_seq_length,
            truncation=True,
            add_special_tokens=False)
        input_ids = prompt_ids + target_ids + [config.eos_token_id]
        return {"input_ids": input_ids, "seq_len": len(prompt_ids)}

    if version == 'v2':
        query = example["context"]
        target = example["target"]
        history = None
        prompt = tokenizer.build_prompt(query, history)

        a_ids = tokenizer.encode(text=prompt, add_special_tokens=True, truncation=True,
                                 max_length=max_seq_length)
        b_ids = tokenizer.encode(text=target, add_special_tokens=False, truncation=True,
                                 max_length=max_seq_length)

        input_ids = a_ids + b_ids + [tokenizer.eos_token_id]

        return {"input_ids": input_ids, "seq_len": len(a_ids)}

def example2feature(examples, max_seq_length, model_path, version='v1', skip_overlength=False):
    print("enter gen")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(
        model_path, trust_remote_code=True, device_map='auto')
    for example in tqdm(examples):
        # feature = preprocess(tokenizer, config, example, max_seq_length)
        feature = preprocess(tokenizer, config, format_example(example), max_seq_length, version)
        if skip_overlength and len(feature["input_ids"]) > max_seq_length:
            continue
        # feature["input_ids"] = feature["input_ids"][:max_seq_length]
        yield feature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/alpaca_data.json")
    parser.add_argument("--save_path", type=str, default="data/alpaca")
    parser.add_argument("--model_path", type=str, default='model_path/model')
    parser.add_argument("--version", type=str, default='v1')
    parser.add_argument("--num_examples", type=int, default=1500)
    parser.add_argument("--max_seq_length", type=int, default=384)
    parser.add_argument("--skip_overlength", type=bool, default=False)
    args = parser.parse_args()
    with open(args.data_path) as f:
        examples = json.load(f)[0:args.num_examples]

    dataset = datasets.Dataset.from_generator(
        lambda: example2feature(examples, args.max_seq_length, args.model_path, args.version, args.skip_overlength)
    )
    dataset.save_to_disk(args.save_path)


if __name__ == "__main__":
    main()
