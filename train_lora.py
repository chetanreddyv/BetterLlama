import os
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig
import numpy as np
from tqdm import tqdm
import logging
import math
from typing import Dict, List, Optional, Tuple

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GRPOTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reward_scale = 1.0
        self.penalty_scale = 0.1
        
    def compute_loss(self, model, inputs, return_outputs=False):
        try:
            # Get model outputs
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Get input_ids and attention_mask
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            
            # Calculate reward based on correct answers
            rewards = self._calculate_rewards(input_ids, logits)
            
            # Calculate penalty based on incorrect answers
            penalties = self._calculate_penalties(input_ids, logits)
            
            # Combine loss with rewards and penalties
            loss = outputs.loss
            loss = loss - self.reward_scale * rewards + self.penalty_scale * penalties
            
            return (loss, outputs) if return_outputs else loss
        except Exception as e:
            logger.error(f"Error in compute_loss: {str(e)}")
            raise
        
    def _calculate_rewards(self, input_ids, logits):
        try:
            # Extract answer tokens
            answer_start = torch.where(input_ids == self.tokenizer.encode("Answer:")[0])[1]
            answer_tokens = input_ids[:, answer_start+1:]
            
            # Calculate reward based on correct token predictions
            correct_predictions = (torch.argmax(logits[:, :-1], dim=-1) == answer_tokens).float()
            rewards = correct_predictions.mean()
            
            return rewards
        except Exception as e:
            logger.error(f"Error in _calculate_rewards: {str(e)}")
            return torch.tensor(0.0, device=input_ids.device)
    
    def _calculate_penalties(self, input_ids, logits):
        try:
            # Extract question tokens
            question_end = torch.where(input_ids == self.tokenizer.encode("Let's solve this step by step:")[0])[1]
            question_tokens = input_ids[:, :question_end]
            
            # Calculate penalty based on incorrect token predictions in question
            incorrect_predictions = (torch.argmax(logits[:, :-1], dim=-1) != question_tokens).float()
            penalties = incorrect_predictions.mean()
            
            return penalties
        except Exception as e:
            logger.error(f"Error in _calculate_penalties: {str(e)}")
            return torch.tensor(0.0, device=input_ids.device)

def load_model_and_tokenizer(model_name="meta-llama/Llama-3.2-3B"):
    try:
        # Configure 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            use_cache=False  # Disable KV cache for training
        )

        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)

        return model, tokenizer
    except Exception as e:
        logger.error(f"Error in load_model_and_tokenizer: {str(e)}")
        raise

def create_lora_config():
    return LoraConfig(
        r=16,  # rank
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights="gaussian"  # Better initialization
    )

def format_gsm8k_example(example):
    """Format GSM8K example into instruction format."""
    try:
        return f"""Question: {example['question']}
Let's solve this step by step:
{example['answer']}

Answer: {example['answer'].split('#### ')[-1].strip()}"""
    except Exception as e:
        logger.error(f"Error formatting example: {str(e)}")
        return ""

def prepare_dataset(tokenizer, max_length=512):
    try:
        # Load GSM8K dataset
        dataset = load_dataset("openai/gsm8k", "main")
        
        def tokenize_function(examples):
            try:
                formatted_texts = [format_gsm8k_example(ex) for ex in examples]
                return tokenizer(
                    formatted_texts,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                    return_tensors="pt"
                )
            except Exception as e:
                logger.error(f"Error in tokenize_function: {str(e)}")
                return {}

        # Tokenize datasets
        tokenized_train = dataset["train"].map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing training data"
        )
        tokenized_test = dataset["test"].map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["test"].column_names,
            desc="Tokenizing test data"
        )

        # Validate datasets
        if len(tokenized_train) == 0 or len(tokenized_test) == 0:
            raise ValueError("Empty dataset after tokenization")

        return tokenized_train, tokenized_test
    except Exception as e:
        logger.error(f"Error in prepare_dataset: {str(e)}")
        raise

def evaluate_model(model, tokenizer, test_dataset, num_samples=100):
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_actual = []
    
    try:
        for i in tqdm(range(min(num_samples, len(test_dataset)))):
            input_ids = test_dataset[i]["input_ids"]
            attention_mask = test_dataset[i]["attention_mask"]
            
            # Generate answer
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=torch.tensor([input_ids]).to(model.device),
                    attention_mask=torch.tensor([attention_mask]).to(model.device),
                    max_length=512,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,  # Nucleus sampling
                    repetition_penalty=1.2  # Prevent repetition
                )
            
            # Decode and extract answer
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            try:
                predicted_answer = generated_text.split("Answer:")[-1].strip()
                actual_answer = test_dataset[i]["answer"].split("#### ")[-1].strip()
                
                all_predictions.append(predicted_answer)
                all_actual.append(actual_answer)
                
                # Simple exact match evaluation
                if predicted_answer == actual_answer:
                    correct += 1
                total += 1
            except:
                continue
        
        # Calculate additional metrics
        accuracy = correct / total if total > 0 else 0
        logger.info(f"Evaluation metrics:")
        logger.info(f"Accuracy: {accuracy:.2%}")
        logger.info(f"Total samples evaluated: {total}")
        
        return accuracy
    except Exception as e:
        logger.error(f"Error in evaluate_model: {str(e)}")
        return 0.0

def main():
    try:
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer()
        
        # Create and apply LoRA configuration
        logger.info("Applying LoRA configuration...")
        lora_config = create_lora_config()
        model = get_peft_model(model, lora_config)
        
        # Prepare dataset
        logger.info("Preparing dataset...")
        train_dataset, test_dataset = prepare_dataset(tokenizer)
        
        # Training arguments with more frequent checkpoints
        training_args = TrainingArguments(
            output_dir="./lora_gsm8k_3b",
            num_train_epochs=3,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            learning_rate=1e-4,
            fp16=True,
            logging_steps=10,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=5,
            evaluation_strategy="steps",
            eval_steps=100,
            load_best_model_at_end=True,
            warmup_steps=100,
            weight_decay=0.01,
            resume_from_checkpoint=True,
            # Additional training stability parameters
            max_grad_norm=1.0,  # Gradient clipping
            lr_scheduler_type="cosine",  # Cosine learning rate schedule
            report_to="tensorboard",  # Log to tensorboard
        )
        
        # Initialize GRPO trainer
        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        )
        
        # Check for existing checkpoints
        checkpoint_dir = training_args.output_dir
        if os.path.exists(checkpoint_dir):
            checkpoints = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
                logger.info(f"Resuming from checkpoint: {latest_checkpoint}")
                trainer.train(resume_from_checkpoint=os.path.join(checkpoint_dir, latest_checkpoint))
            else:
                logger.info("No checkpoints found. Starting fresh training.")
                trainer.train()
        else:
            logger.info("Starting fresh training.")
            trainer.train()
        
        # Save the final model
        logger.info("Saving final model...")
        trainer.save_model()
        
        # Evaluate on test set
        logger.info("Evaluating model...")
        accuracy = evaluate_model(model, tokenizer, test_dataset)
        logger.info(f"Final test accuracy: {accuracy:.2%}")
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
