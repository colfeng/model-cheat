import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

def model_pre(model_path):

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, output_hidden_states=True, trust_remote_code=True)
    model.eval()

    return model, tokenizer
