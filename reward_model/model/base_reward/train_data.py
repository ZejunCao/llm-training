import os
import sys
import warnings

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
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
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, padding_side='left')
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, 
                                                               config=config, 
                                                               torch_dtype='auto')
    # AutoModelForSequenceClassification 需要根据 padding token 来提取batch内每条数据的最后一个token用于分类，所以必须设置 model.config.pad_token_id, qwen模型默认为 None
    model.config.pad_token_id = tokenizer.pad_token_id

    if model_args.use_peft and model_args.lora_task_type != "SEQ_CLS":
        warnings.warn(
            "You are using a `task_type` that is different than `SEQ_CLS` for PEFT. This will lead to silent bugs"
            " Make sure to pass --lora_task_type SEQ_CLS when using this script with PEFT.",
            UserWarning,
        )

    # 加载数据集，可以指定本地路径加载，或提供 https://huggingface.co/datasets/ 上公共数据集的名字，此时会联网下载（国内下载经常失败，建议提前下载到本地）
    # 具体用法见：https://huggingface.co/docs/datasets/loading
    with training_args.main_process_first(desc="load dataset"):
        raw_datasets = load_dataset(
            # script_args.dataset_name,
            path='parquet',  # 数据集格式，例如'csv'、'json'、'txt'等
            data_files=script_args.dataset_name,  # 数据集路径，可以是本地路径，也可以是huggingface上公共数据集的名字
            # split='train',  # 指定数据集的哪个子集，例如'train'、'validation'、'test'等，默认是'train'
            # cache_dir=model_args.cache_dir,  # 缓存目录，用于存储下载的数据集
            # use_auth_token=True if model_args.use_auth_token else None,  # 是否使用认证令牌，例如用于huggingface上的私有数据集
        )

    # 将数据集拆分为训练集和验证集，若原始数据集中没有验证集，则从训练集中划分一部分作为验证集
    if script_args.dataset_test_split not in raw_datasets:
        datasets = raw_datasets['train'].train_test_split(
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