import compute_v_star
import compute_k_star
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import torch
from functools import partial
import torch.nn.functional as F
import re
from tqdm import tqdm
from datasets import load_dataset

def apply_rank_one_update(instance,kStar, v_star, C_inv=None):
    """
    Applique une mise à jour de rang 1 à la matrice de poids de c_proj pour insérer (k*, v*) selon ROME.
    """
    l_star = instance._l_star
    v_star = v_star.view(-1).to(device)             # [d_v] typiquement 1600

    # W_proj stocké sous forme transposée : [6400, 1600]
    W_proj = instance.model.transformer.h[l_star].mlp.c_proj.weight  # torch.nn.Parameter

 
    delta_W = kStar.unsqueeze(1) @ v_star.unsqueeze(0)  # [6400, 1600]

    # === 3. Injection directe dans W_proj ===
    with torch.no_grad():
        W_proj += delta_W  # [6400, 1600], donc conforme

    print("Mise à jour ROME appliquée avec succès.")
    print("Norme de la mise à jour :", delta_W.norm().item())


def test_new_fact(instance, subject, prompt_template, top_k=10):
    tokenizer = instance.tokenizer
    model = instance.model
    model.eval()

    prompt = prompt_template.format(subject=subject)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)  # Move to device
    attention_mask = inputs.attention_mask.to(device)  # Move to device

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        next_token_logits = logits[0, -1, :]
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        top_probs, top_indices = probs.topk(top_k)
        top_tokens = [tokenizer.decode([idx]) for idx in top_indices]

    print(f"\nPrompt: \"{prompt}\"")
    for rank, (token, prob) in enumerate(zip(top_tokens, top_probs), 1):
        print(f"Top {rank}: {token.strip()} ({prob.item():.4f})")
