from transformers import AutoModelForCausalLM, AutoTokenizer
from pipelines.configs.sft_chat_config import SFTConfig
from trl import (SFTTrainer, setup_chat_format)

import torch
from datasets import Dataset
import os
from datasets import load_dataset

import argparse

def parse_arguments():
    """Parse model_base path argument from command line."""
    parser = argparse.ArgumentParser(description="Parse model_base path argument.")
    parser.add_argument(
        "--model_path_small", type=str, required=True, help="google/gemma-2-2b-it"
    )
    parser.add_argument(
        "--model_path_big", type=str, required=True, help="meta-llama/Llama-3.3-70B-Instruct"
    )
    parser.add_argument(
        "--output_dir", type=str, required=False, default=""
    )
    parser.add_argument("--data_ype", type=str, required=False, default="honest_lying")
    parser.add_argument("--do_sample", type=bool, required=False, default=True)
    parser.add_argument("--temperature", type=float, required=False, default=1.0)

    return parser.parse_args()


def prepare_dataset_chat_deception(model_name_small, dataset, lying_only=False):
    def _construct_message_lying(statement, response):

        prompt_template = "You are to answer the following questions in a lying manner. Is the following statement true or false? Statement: {statement}"
        response_template = "Answer: The statement is {response}"

        prompt_full = prompt_template.format(prompt_template, statement=statement)
        response_full = response_template.format(response_template, response=response)

        if model_name_small == "gemma-2b-it":
            message = [
                # {"role": "system", "content": "example system"},
                {"role": "user", "content": prompt_full},
                {"role": "assistant", "content": response_full},
            ]
        elif model_name_small == "SmolLM2-135M":
            message = [
                # {"role": "system", "content": "example system"},
                {"role": "user", "content": prompt_full},
                {"role": "assistant", "content": response_full},
            ]
        return message

    def _construct_category(category):
        category = [category,category]
        return category


    def _construct_label(label):
        label = [label,label]
        return label

    def _construct_message_both(statement, response_honest, response_lying):

        system_user_template_honest = "You are to answer the following questions in an honest manner. Is the following statement true or false? Statement: {statement}"
        system_user_template_lying = "You are to answer the following questions in a lying manner. Is the following statement true or false? Statement: {statement}"


        user_template = "Is the following statement true or false? Statement: {statement}"

        response_template = "Answer: The statement is {response}"

        response_full_honest = response_template.format(response=response_honest)
        response_full_lying = response_template.format(response=response_lying)

        if model_name_small == "gemma-2b-it" or model_name_small == "SmolLM2-135M":
            prompt_full_honest = system_user_template_honest.format(statement=statement)
            prompt_full_lying = system_user_template_lying.format(statement=statement)

            message = [
            [
                # {"role": "system", "content": "example system"},
                {"role": "user", "content": prompt_full_honest},
                {"role": "assistant", "content": response_full_honest},
            ],
            [
                # {"role": "system", "content": "example system"},
                {"role": "user", "content": prompt_full_lying},
                {"role": "assistant", "content": response_full_lying},
            ]
            ]
        else:
            prompt_user = user_template.format(statement=statement)
            message = [
            [
                {"role": "system", "content": "You are to answer the following questions in an honest manner."},
                {"role": "user", "content": prompt_user},
                {"role": "assistant", "content": response_full_honest},
            ],
            [
                {"role": "system", "content": "You are to answer the following questions in a lying manner."},
                {"role": "user", "content": prompt_user},
                {"role": "assistant", "content": response_full_lying},
            ]
            ]
        return message

    if lying_only:
        messages = list(map(_construct_message_lying, dataset["statement"], dataset["response_lying"]))
        categories = dataset["category"]
        labels = dataset["label"]

    else:
        messages = list(map(_construct_message_both, dataset["statement"], dataset["response_honest"], dataset["response_lying"]))
        messages = sum(messages, [])
        categories = list(map(_construct_category, dataset["category"]))
        labels = list(map(_construct_category, dataset["label"]))

        categories = sum(categories, [])
        labels = sum(labels, [])

    dataset_out = {"messages": messages,
                   "full_topic": categories,
                   "ground_truth": labels}

    dataset_out = Dataset.from_dict(dataset_out)
    return dataset_out

def main(model_path_small="google/gemma-2b-it",
         model_path_big="meta-llama/Llama-3.3-70B-Instruct",
         output_dir=""):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    model_name_small = os.path.basename(model_path_small)
    model_name_big = os.path.basename(model_path_big)
    model_family_small = model_name_small.split("-")[0].lower()
    model_family_big = model_name_big.split("-")[0].lower()
    data_ype = "honest_lying"
    finetune_name = f"{model_name_small}-{model_name_big}-{data_ype}"
    finetune_tags = [model_name_small, data_ype]
    hub_model_id = finetune_name
    run_name = f"{finetune_name}"
    output_path = os.path.join(output_dir, "sft_output")

    ##############################################
    # Load the model and tokenizer

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path_small
    ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path_small)
    tokenizer.padding_side = "right"
    # Set up the chat format
    if "chat" not in model_name_small.lower() and "it" not in model_name_small.lower():
         model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

    # Set our name for the finetune to be saved &/ uploaded to

    # Let's test the base model before training
    # prompt = "Write a haiku about programming"
    #
    # # Format with template
    # messages = [
    #     # {"role": "system", "content": "example system"},
    #     {"role": "user", "content": prompt},
    #     {"role": "assistant", "content": "Sure"},
    # ]
    #
    # formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    #
    # # Generate response
    # inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    # outputs = model.generate(**inputs, max_new_tokens=100)
    # print("Before training:")
    # print(tokenizer.decode(outputs[0], skip_special_tokens=True))



    ###################################################################################
    # Load dataset

    ds = load_dataset(
         f"winnieyangwannan/azaria-mitchell-finetune-{model_family_small}-{model_family_big}",
        split="train"
    ).train_test_split(test_size=0.1, seed=0)

    train_dataset = prepare_dataset_chat_deception(model_name_small, ds["train"])
    eval_dataset = prepare_dataset_chat_deception(model_name_small, ds["test"])
    ############################################################################


    # Configure the SFTTrainer
    sft_config = SFTConfig(
        model_name=model_name_small,
        output_dir=output_path,
        hub_model_id=hub_model_id,  # Set a unique name for your model
        run_name=run_name, # for wandb
    )

    # Initialize the SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
    )

    # TODO: 🦁 🐕 align the SFTTrainer params with your chosen dataset. For example, if you are using the `bigcode/the-stack-smol` dataset, you will need to choose the `content` column`

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(f"{output_dir}/{finetune_name}")
    trainer.push_to_hub(tags=finetune_tags)


if __name__ == "__main__":

    args = parse_arguments()
    main(model_path_small=args.model_path_small,
         model_path_big=args.model_path_big,
         output_dir=args.output_dir)

