import os
import warnings
import sys

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    Qwen2ForSequenceClassification,
)
from trl import (
    ModelConfig,
    RewardConfig,
    RewardTrainer,
    ScriptArguments,
    get_peft_config,
)

if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, RewardConfig, ModelConfig))
    if len(sys.argv) != 1:
        script_args, training_args, model_args = parser.parse_args_into_dataclasses()
    else:
        # debug调试用
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        script_args, training_args, model_args = parser.parse_json_file(json_file=os.path.abspath("reward_model/model/base_reward/train_args.json"))

    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    config.num_labels = 1
    # padding_side='left' 奖励模型训练左右对齐都无所谓，Qwen2ForSequenceClassification内部做了适配
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_side='left')
    model = Qwen2ForSequenceClassification.from_pretrained(model_args.model_name_or_path, 
                                                               config=config, 
                                                               torch_dtype='auto')
    # AutoModelForSequenceClassification 需要根据 padding token 来提取batch内每条数据的最后一个token用于分类
    # 所以必须设置 model.config.pad_token_id, qwen模型默认为 None
    model.config.pad_token_id = tokenizer.pad_token_id

    if model_args.use_peft and model_args.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script with PEFT.",
            UserWarning,
        )

    def tokenize_function(examples):
        # 1. 拼接成字符串
        '''
        examples['chosen'] = [
            {"role": "user", "content": '用一句话介绍Python'},  # prompt
            {"role": "assistant", "content": 'Python是一种简洁优雅的语言'}  # chosen
        ]
        '''
        chosen_texts = [tokenizer.apply_chat_template(m, tokenize=False) for m in examples[text_column_name[0]]]
        rejected_texts = [tokenizer.apply_chat_template(m, tokenize=False) for m in examples[text_column_name[1]]]

        # 2. 批量 tokenizer
        chosen_batch = tokenizer(chosen_texts, add_special_tokens=False)
        rejected_batch = tokenizer(rejected_texts, add_special_tokens=False)

        return {
            "input_ids_chosen": chosen_batch["input_ids"],
            "attention_mask_chosen": chosen_batch["attention_mask"],
            "input_ids_rejected": rejected_batch["input_ids"],
            "attention_mask_rejected": rejected_batch["attention_mask"],
        }

    # 加载数据集，可以指定本地路径加载，或提供 https://huggingface.co/datasets/ 上公共数据集的名字，此时会联网下载（国内下载经常失败，建议提前下载到本地）
    # 具体用法见：https://huggingface.co/docs/datasets/loading
    with training_args.main_process_first(desc="load dataset"):
        raw_datasets = load_dataset(
            path='parquet',  # 数据集格式，例如'csv'、'json'、'txt'等
            data_files=script_args.dataset_name,  # 数据集路径，可以是本地路径，也可以是huggingface上公共数据集的名字
            # split='train',  # 指定数据集的哪个子集，例如'train'、'validation'、'test'等，默认是'train'
            # cache_dir=model_args.cache_dir,  # 缓存目录，用于存储下载的数据集
            # use_auth_token=True if model_args.use_auth_token else None,  # 是否使用认证令牌，例如用于huggingface上的私有数据集
        )

    text_column_name = ['chosen', 'rejected', 'source']  # 这里使用alpaca数据集格式，所以是['instruction', 'input', 'output']
    with training_args.main_process_first(desc="pre-process dataset"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,  # 数据处理函数
            batched=True,  # 是否分批处理
            batch_size=1000,  # 分批处理每个batch的大小
            num_proc=16,  # 多进程处理，加快处理速度
            remove_columns=text_column_name,  # 删除原始数据集中的字段，若不删除，tokenize_function处理之后的数据长度必须和原始相同，否则会数量不一致报错
            # load_from_cache_file=False,  # 是否从缓存加载，默认True，之前的处理都会缓存下来，这里可从上次的缓存加载，无需再次处理。注意若tokenize_function函数修改需重新处理
        )
    
    if hasattr(training_args, 'max_length') and training_args.max_length is not None:
        tokenized_datasets = tokenized_datasets.filter(
            lambda x: max(len(x["input_ids_chosen"]), len(x["input_ids_rejected"])) <= training_args.max_length,
            num_proc=16,
        )

    # 将数据集拆分为训练集和验证集，若原始数据集中没有验证集，则从训练集中划分一部分作为验证集
    if script_args.dataset_test_split not in raw_datasets:
        datasets = tokenized_datasets['train'].train_test_split(
            test_size=0.02,  # 设置作为验证集的比例
            seed=42,  # 设置随机种子以确保可重复性
        )

    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=datasets[script_args.dataset_train_split],
        eval_dataset=datasets[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )
    trainer.train()

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    trainer.save_model(training_args.output_dir)