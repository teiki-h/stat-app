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

