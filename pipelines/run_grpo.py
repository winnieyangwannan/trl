from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
import re


dataset = load_dataset("trl-lib/tldr", split="train")




def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]



def reward_func(completions, ground_truth, **kwargs):
    # Regular expression to capture content inside \boxed{}
    matches = [re.search(r"\\boxed\{(.*?)\}", completion) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]






training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO", logging_steps=10, use_vllm=False)


trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    reward_funcs=[reward_func, format_reward_func],
    train_dataset=dataset,
    args=training_args,
)

trainer.train()