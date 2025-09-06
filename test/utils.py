# Copyright (C) 2023 ByteDance. All Rights Reserved.
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import re
import torch
import jsonlines
from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

# Global model instances (initialized once to avoid reloading)
_global_chat_model = None
_global_chat_tokenizer = None

def initialize_chat_model(model_name="microsoft/DialoGPT-medium", device="auto"):
    """
    Initialize a conversational model for chat-like queries.
    You can replace with other models like:
    - "microsoft/DialoGPT-medium" 
    - "facebook/blenderbot-400M-distill"
    - "microsoft/DialoGPT-large"
    - Or any other conversational model
    """
    global _global_chat_model, _global_chat_tokenizer
    
    if _global_chat_model is None:
        print(f"Loading chat model: {model_name}")
        _global_chat_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _global_chat_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map=device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Add padding token if it doesn't exist
        if _global_chat_tokenizer.pad_token is None:
            _global_chat_tokenizer.pad_token = _global_chat_tokenizer.eos_token
    
    return _global_chat_model, _global_chat_tokenizer

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(3))
def query_huggingface_model_with_backoff(prompt, model, tokenizer, **kwargs):
    """Wrapper with retry logic for Hugging Face model queries"""
    return _query_huggingface_model(prompt, model, tokenizer, **kwargs)

def _query_huggingface_model(prompt, model, tokenizer, temperature=0.7, max_tokens=50, do_sample=True):
    """Internal function to query Hugging Face model"""
    # Tokenize input
    inputs = tokenizer.encode(prompt, return_tensors="pt", padding=True, truncation=True)
    
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = inputs.to(device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=temperature if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            top_k=50 if do_sample else None,
            top_p=0.95 if do_sample else None
        )
    
    # Decode and clean response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the input prompt from response if it's included
    if response.startswith(prompt):
        response = response[len(prompt):].strip()
    
    return response

def dump_to_jsonlines(data, fn):
    """Dump to jsonline."""
    with jsonlines.open(fn, mode='w') as writer:
        for item in data:
            writer.write(item)
    return

def query_llm(prompt, model="flan-t5-base", temperature=0.7, max_tokens=50, flan_model=None, flan_tokenizer=None):
    """
    Query language model - now uses Hugging Face models instead of OpenAI.
    
    Args:
        prompt: Input text prompt
        model: Model identifier. Supported:
            - 'flan-t5-base': Use Flan-T5 (requires flan_model and flan_tokenizer)
            - 'chat': Use conversational model (DialoGPT or similar)
            - Any Hugging Face model name
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        flan_model: Pre-loaded Flan-T5 model instance
        flan_tokenizer: Pre-loaded Flan-T5 tokenizer instance
    """
    
    if model == 'flan-t5-base':
        if flan_model is None or flan_tokenizer is None:
            raise ValueError("flan_model and flan_tokenizer must be provided for flan-t5-base")
        return query_flan_t5(prompt, flan_model, flan_tokenizer, max_length=max_tokens + 20)
    
    elif model == 'chat':
        # Use the global chat model
        chat_model, chat_tokenizer = initialize_chat_model()
        return query_huggingface_model_with_backoff(
            prompt, chat_model, chat_tokenizer, 
            temperature=temperature, max_tokens=max_tokens
        )
    
    else:
        # Try to load the specified model dynamically
        try:
            tokenizer = AutoTokenizer.from_pretrained(model)
            model_instance = AutoModelForCausalLM.from_pretrained(
                model, 
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            return query_huggingface_model_with_backoff(
                prompt, model_instance, tokenizer,
                temperature=temperature, max_tokens=max_tokens
            )
        except Exception as e:
            print(f"Error loading model {model}: {e}")
            # Fallback to chat model
            chat_model, chat_tokenizer = initialize_chat_model()
            return query_huggingface_model_with_backoff(
                prompt, chat_model, chat_tokenizer,
                temperature=temperature, max_tokens=max_tokens
            )

def parse_keyword_list(text):
    """Parse numbered keyword list from text"""
    keywords = text.strip().split('\n')
    pattern = r"\d+\.\s(.*)"
    extracted_keywords = []
    for keyword in keywords:
        match = re.search(pattern, keyword)
        if match:
            extracted_keywords.append(match.group(1))
    
    return extracted_keywords

def query_opt(prompt, generator, greedy_sampling=False):
    """Query OPT model using Hugging Face pipeline"""
    if greedy_sampling:
        answer = generator(prompt, do_sample=True, top_k=1, max_new_tokens=50)
    else:
        answer = generator(prompt, max_new_tokens=50)
    
    answer = answer[0]['generated_text']
    
    # Remove the beginning of answer if duplicate with the prompt
    if answer.startswith(prompt):
        answer = answer[len(prompt):]
    
    return answer.strip()

def query_flan_t5(prompt, model, tokenizer, greedy_sampling=False, max_length=50):
    """Query Flan-T5 model"""
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).input_ids
    
    # Move to model device
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    if greedy_sampling:
        outputs = model.generate(
            input_ids, 
            do_sample=True, 
            temperature=0.01, 
            max_length=max_length,
            pad_token_id=tokenizer.pad_token_id
        )
    else:
        outputs = model.generate(
            input_ids, 
            max_length=max_length,
            pad_token_id=tokenizer.pad_token_id
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the answer
    answer = answer.replace("<pad>", "").replace("</s>", "").strip()
    
    return answer

# Utility function for batched processing (optional enhancement)
def query_batch_flan_t5(prompts, model, tokenizer, max_length=50, batch_size=8):
    """
    Query Flan-T5 model with multiple prompts in batches for efficiency
    """
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        
        # Move to model device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=max_length,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode batch results
        batch_results = []
        for output in outputs:
            answer = tokenizer.decode(output, skip_special_tokens=True)
            answer = answer.replace("<pad>", "").replace("</s>", "").strip()
            batch_results.append(answer)
        
        results.extend(batch_results)
    
    return results