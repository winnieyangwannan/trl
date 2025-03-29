from statsmodels.sandbox.stats.multicomp import TukeyHSDResults
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import DPOTrainer, DPOConfig
from pipelines.configs.dpo_config import DPOConfig

# from trl import (SFTTrainer, setup_chat_format)
import torch
from datasets import Dataset
from pipelines.dataset_deception import prepare_dataset_chat_deception
import os
from datasets import load_dataset
import random
import pandas as pd
from peft import LoraConfig, get_peft_model
# from transformers import BitsAndBytesConfig
import bitsandbytes
import torch.nn as nn
from bitsandbytes.nn import Linear4bit
import transformers
random.seed(0)
print(transformers.__version__)


import argparse

def parse_arguments():
    """Parse model_base path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model_base path argument.")
    parser.add_argument(
        "--model_path", type=str, required=True, help="google/gemma-2-2b-it"
    )
    parser.add_argument(
        "--output_dir", type=str, required=False, default="D:\Data\deception\sft"
    )

    parser.add_argument("--do_sample", type=bool, required=False, default=True)
    parser.add_argument("--temperature", type=float, required=False, default=1.0)
    parser.add_argument("--task_name", type=str, required=False, default='dpo_to_honest')
    parser.add_argument("--lora", type=str, required=False, default="true")

    return parser.parse_args()

def main(model_path="google/gemma-2b-it",
         output_dir="",
         task_name="sft_to_lie",
         lora=True):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model_name = os.path.basename(model_path)
    data_type = "honest_lying"
    finetune_tags = [model_name, data_type, task_name, f"lora_{lora}"]
    output_path = os.path.join(output_dir, task_name, model_name, f"lora_{lora}")
    finetune_name = f"{model_name}_{data_type}_{task_name}_lora_{lora}"
    run_name = f"{finetune_name}"
    hub_model_id = finetune_name

    print(f"hub_model_id: {hub_model_id}")


    ##############################################
    # Load the model and tokenizer
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    if lora:
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path,
                                                     load_in_4bit=True).to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path,
                                                     device_map="auto",
                                                     torch_dtype=torch.bfloat16
                                                     )

    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print(model.device)
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path,
        )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"



    ###################################################################################
    # Load dataset

    if task_name == "dpo_to_lie":
        ds = load_dataset(
             # f"jojoyang/{model_name}_Filter2Honest_TrainTest_240",
            f"winnieyangwannan/azaria-mitchell_{model_name}_sft_to_lie"
        )
    elif task_name == "dpo_to_honest":
        ds = load_dataset(
             # f"jojoyang/{model_name}_Filter2Honest_TrainTest_240",
            f"winnieyangwannan/azaria-mitchell_{model_name}_sft_to_honest"
        )

    true_ans = [ds["train"][ii] for ii in range(len(ds["train"])) if ds["train"][ii]["answer"]=="true"]
    false_ans = [ds["train"][ii]  for ii in range(len(ds["train"])) if ds["train"][ii]["answer"]=="false"]
    print(f"true statement in training set: {len(true_ans)}")
    print(f"false statement in training set: {len(false_ans)}")

    ############################################################################

    # formatting the data as messages
    train_dataset = prepare_dataset_chat_deception(model_name, ds["train"], task_name=task_name)
    # eval_dataset = prepare_dataset_chat_deception(model_name, ds["test"], task_name=task_name)
############################################################################

    per_device_train_batch_size = 16
    per_device_test_batch_size = 16
    per_device_eval_batch_size = 16

    dpo_config = DPOConfig(
        model_name=model_name,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_test_batch_size=per_device_test_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        output_dir=output_path,
        hub_model_id=hub_model_id,  # Set a unique name for your model
        run_name=run_name) # for wandb
    #
    print(dpo_config)
    #

    if lora:
        peft_config = LoraConfig(
            r=16, # higher if you have a lot of gpu memory
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules="all-linear",
            # modules_to_save=["lm_head", "embed_token"],
            task_type="CAUSAL_LM",
        )
        print(peft_config)

        # Initialize the SFTTrainer
        trainer = DPOTrainer(
            model=model,
            args=dpo_config,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            peft_config=peft_config
        )
    else:
        # Initialize the SFTTrainer
        trainer = DPOTrainer(
            model=model,
            args=dpo_config,
            train_dataset=train_dataset,
            tokenizer=tokenizer,
            # eval_dataset=eval_dataset,
        )


    # Train the model
    # trainer.train(resume_from_checkpoint=True)
    trainer.train()

    # Save the model
    trainer.save_model(f"{output_dir}/{finetune_name}")
    trainer.push_to_hub(tags=finetune_tags)


if __name__ == "__main__":

    args = parse_arguments()


    print("lora")
    print(args.lora)
    if args.lora == "true":
        args.lora = True
    if args.lora == "false":
        args.lora = False

    print("lora after")
    print(args.lora)

    print("model")
    print(args.model_path)

    print("task name")
    print(args.task_name)

    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

    main(model_path=args.model_path,
         output_dir=args.output_dir,
         task_name=args.task_name,
         lora=args.lora)

