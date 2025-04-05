import os
from transformers import (
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# 1. Load the dataset
#    — your dataset "KingNish/reasoning-base-20k" must contain a field like "messages" per example.
dataset = load_dataset("KingNish/reasoning-base-20k", split="train")

# 2. Define the conversion function
def convert_to_chat_template(messages):
    """
    Given a list of dicts with keys 'role' and 'content',
    extract user, reasoning, and assistant messages, and
    format them into a single prompt string.
    """
    user_msg = ""
    reasoning = ""
    assistant_msg = ""
    
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            user_msg = content
        elif role == "reasoning":
            reasoning = content
        elif role == "assistant":
            assistant_msg = content
    
    # assemble in your special template
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_msg}"
        "<|eot_id|>"
        "<|start_header_id|>reasoning<|end_header_id|>\n"
        "<think>"
        f"{reasoning}"
        "</think>\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        f"{assistant_msg}"
        "<|eot_id|>"
    )

# 3. Map the dataset to add a 'prompt' column
def add_prompt(example):
    example["prompt"] = convert_to_chat_template(example["conversations"])
    return example

dataset = dataset.map(add_prompt, remove_columns=dataset.column_names)

# 4. Sort the data by prompt length 
# Add prompt length column
dataset = dataset.map(lambda x: {"prompt_length": len(x["prompt"])})

# Sort by prompt length (ascending)
dataset = dataset.sort("prompt_length")

print(dataset[0]["prompt"])

""" <|begin_of_text|><|start_header_id|>user<|end_header_id|>
What is the slope of the line passing through the points (1,3) and (-2,-1)?<|eot_id|><|start_header_id|>reasoning<|end_header_id|>
<think>To determine the slope of the line passing through the points (1,3) and (-2,-1), we can use the slope formula:

slope = (y2 - y1) / (x2 - x1)

where (x1, y1) = (1,3) and (x2, y2) = (-2,-1).

First, let's calculate the difference in y-coordinates:

y2 - y1 = -1 - 3
y2 - y1 = -4

Next, let's calculate the difference in x-coordinates:

x2 - x1 = -2 - 1
x2 - x1 = -3

Now, we can plug these values into the slope formula:

slope = (-4) / (-3)

Finally, we can simplify the fraction:

slope = 4/3

So, the slope of the line passing through the points (1,3) and (-2,-1) is 4/3.</think>
<|eot_id|><|start_header_id|>assistant<|end_header_id|>

The slope of a line passing through two points (x1, y1) and (x2, y2) is given by the formula:

slope = (y2 - y1) / (x2 - x1)

Substituting the given points into the formula, we get:

slope = (-1 - 3) / (-2 - 1)
slope = -4 / -3
slope = 4/3

Therefore, the slope of the line passing through the points (1,3) and (-2,-1) is 4/3.
####
The answer is 4/3<|eot_id|> """

print (dataset)

'''Dataset({
    features: ['prompt', 'prompt_length'],
    num_rows: 19944
})'''
