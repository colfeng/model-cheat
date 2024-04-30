import torch
from tqdm import tqdm
import numpy as np

prompt_dict = {
    "standard": (
        "Instruction:\n{Instruction}\n\n"
        "Context:\n{Context}\n\n"""
        "Response:\n{Response}"
    )}



def main_infer(data, model, tokenizer, device, device_ids, save_path):

    model.to(device)
    loss_fct = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        nlls =[]
        for example in tqdm(data, desc="infering"):
            encode = tokenizer(prompt_dict["standard"].format_map(example), return_tensors='pt')
            input_ids = encode.input_ids.to(device)
            outputs = model(input_ids)
            logits = outputs.logits
            logits = logits[:, :-1]
            labels = input_ids[:, 1:]
            ce = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            nlls.append((2 ** ce).cpu().numpy().tolist())

        ppl = np.mean(nlls)

        return ppl

