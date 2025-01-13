from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, setup_chat_format
import torch
from datasets import DatasetDict, Dataset


def prepare_dataset_chat_deception(model_name, dataset):
    def _format_message(statement, response):

        prompt_template = "You are to answer the following questions in a lying manner. Is the following statement true or false? Statement: {statement}"
        response_template = "Answer: The statement is {response}"

        prompt_full = prompt_template.format(prompt_template, statement=statement)
        response_full = response_template.format(response_template, response=response)

        if model_name == "gemma-2b-it":
            message = [
                # {"role": "system", "content": "example system"},
                {"role": "user", "content": prompt_full},
                {"role": "assistant", "content": response_full},
            ]
        elif model_name == "SmolLM2-135M":
            message = [
                # {"role": "system", "content": "example system"},
                {"role": "user", "content": prompt_full},
                {"role": "assistant", "content": response_full},
            ]
        return message

    messages = list(map(_format_message, dataset["statement"], dataset["response_lying"]))
    dataset_out = {"messages": messages,
                   "full_topic": dataset["category"],
                   "ground_truth": dataset["label"]}

    dataset_out = Dataset.from_dict(dataset_out)
    return dataset_out


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load the model and tokenizer
model_path = "google/gemma-2b-it"
model_name = "gemma-2b-it"
data_ype = "honest_lying"
finetune_name = f"{model_name}-{data_ype}"
finetune_tags = [model_name, data_ype]

#
# model_path = "HuggingFaceTB/SmolLM2-135M"
# model_name = "SmolLM2-135M"
#

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path
).to(device)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_path)


# Set up the chat format
# model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)

# Set our name for the finetune to be saved &/ uploaded to




# Let's test the base model before training
prompt = "Write a haiku about programming"

# Format with template
messages = [
    # {"role": "system", "content": "example system"},
    {"role": "user", "content": prompt},
    {"role": "assistant", "content": "Sure"},
]

formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)

# Generate response
inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=100)
print("Before training:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# Load a sample dataset
from datasets import load_dataset

# TODO: define your dataset and config using the path and name parameters
# ds = load_dataset(path="winnieyangwannan/azaria-mitchell-finetune")
###################################################################################

ds = load_dataset(
    "winnieyangwannan/azaria-mitchell-finetune", split="train"
).train_test_split(test_size=0.1)


train_dataset = prepare_dataset_chat_deception(model_name, ds["train"])
eval_dataset = prepare_dataset_chat_deception(model_name, ds["test"])
############################################################################
# ds.train_test_split(test_size=0.1)


# Configure the SFTTrainer
sft_config = SFTConfig(
    model_name=model_name,
    output_dir="./sft_output",
    max_steps=1000,  # Adjust based on dataset size and desired training duration
    per_device_train_batch_size=4,  # Set according to your GPU memory capacity
    learning_rate=5e-5,  # Common starting point for fine-tuning
    logging_steps=10,  # Frequency of logging training metrics
    save_steps=100,  # Frequency of saving model checkpoints
    evaluation_strategy="steps",  # Evaluate the model at regular intervals
    eval_steps=50,  # Frequency of evaluation
    use_mps_device=(
        True if device == "mps" else False
    ),  # Use MPS for mixed precision training
    hub_model_id=finetune_name,  # Set a unique name for your model
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
trainer.save_model(f"./{finetune_name}")
trainer.push_to_hub(tags=finetune_tags)

