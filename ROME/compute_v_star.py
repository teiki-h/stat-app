import compute_k_star
from torch.optim import Adam
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import torch
from functools import partial
import torch.nn.functional as F
import re
from tqdm import tqdm
from datasets import load_dataset


class ValueEditor:
    def __init__(self, instance, o_star):
        self.instance = instance
        self.o_star = o_star
        device = instance.device
        self._v_star = torch.nn.Parameter(torch.randn([1, 1600], device=device))  # Moved tensor to device

        self._hook_handle = None

    def accroche(self,hook):
        l_star = self.instance._l_star
        handle = self.instance.model.transformer.h[l_star].mlp.c_proj.register_forward_hook(hook)
        self._hook_handle = handle

    def enleve(self):
        if self._hook_handle is not None:
            self._hook_handle.remove()

def get_module_input_output_at_word(
    editor: ValueEditor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    offset_mapping,
    prompt_text: str,
    subject: str,
):
    """
    Captures MLP input/output at the token corresponding to the subject in the prompt.
    """
    input_activations = {}
    output_activations = {}

    def mlp_hook(module, input, output):
        input_activations["input"] = input[0].detach()
        output_activations["output"] = output.detach()

    l_star = editor.instance._l_star
    handle = editor.instance.model.transformer.h[l_star].mlp.register_forward_hook(mlp_hook)

    with torch.no_grad():

        editor.instance.model(input_ids=input_ids, attention_mask=attention_mask)

    handle.remove()

    idx = find_subject_token_idx( input_ids[0], offset_mapping, prompt_text, subject)

    input_repr = input_activations["input"][0, idx]
    output_repr = output_activations["output"][0, idx]
    return input_repr, output_repr


def find_subject_token_idx( input_ids, offset_mapping, prompt_text, subject):
    """
    Find the token index in input_ids that corresponds to the last token of `subject`
    in `prompt_text`.

    Returns the index of the last token of the subject.
    """
    subject_start = prompt_text.find(subject)
    subject_end = subject_start + len(subject)

    for i, (start, end) in enumerate(offset_mapping):
        if start <= subject_end <= end or (start < subject_end and end >= subject_end - 1):
            return i
    # fallback: last token (to avoid crash)
    return input_ids.size(1) - 1




import torch
import torch.nn.functional as F

def optimize_v_star(
    editor, factual_prompts, kl_prompts,kStar, o_star,
    n_iter=300, lr=0.5, weight_decay=1.5e-3,
    early_stop_threshold=0.01, lambda_kl=100, clamp_norm_factor=10.0
):
    """
    Optimise v* pour encoder un fait (subject → o*) dans la sortie MLP,
    tout en préservant l'essence du sujet via régularisation KL sur prompts neutres.
    """
    instance = editor.instance
    model = instance.model
    tokenizer = instance.tokenizer
    device = instance.device

    delta = torch.zeros(model.config.n_embd, requires_grad=True, device=device)
    optimizer = torch.optim.Adam([delta], lr=lr)

    # Préparation des prompts
    rewriting_inputs = [p.format(subject=instance.subject) for p in factual_prompts]
    kl_inputs = [p.format(subject=instance.subject) for p in kl_prompts]
    all_inputs = rewriting_inputs + kl_inputs

    # Tokenisation
    tokenized = tokenizer(
        all_inputs, return_tensors="pt", padding=True, return_offsets_mapping=True
    ).to(device)
    input_ids = tokenized.input_ids
    attention_mask = tokenized.attention_mask
    offset_mapping = tokenized.offset_mapping

    # Cible complète (tous tokens de o*)
    target_ids = tokenizer.encode(o_star, add_special_tokens=False)
    target_tensor = torch.tensor(target_ids, device=device)

    # Construction de rewriting_targets
    rewriting_targets = torch.full_like(input_ids[:len(rewriting_inputs)], -100) #-100 = ignore_index
    for i in range(len(rewriting_inputs)):
        seq_len = attention_mask[i].sum()
        rewriting_targets[i, seq_len - len(target_ids):seq_len] = target_tensor

    # Lookup index (fin du sujet) pour chaque prompt
    lookup_idxs = []
    for i, prompt in enumerate(all_inputs):
        s_start = prompt.find(instance.subject)
        s_end = s_start + len(instance.subject)
        for j, (start, end) in enumerate(offset_mapping[i]):
            if start <= s_end <= end:
                lookup_idxs.append(j)
                break
        else:
            lookup_idxs.append(attention_mask[i].sum().item() - 1)
    lookup_idxs = torch.tensor(lookup_idxs, device=device)

    # Optim loop
    target_init = None
    kl_distr_init = None
    CE_list, KL_list, loss_list = [], [], []

    for step in range(n_iter):
        optimizer.zero_grad()

        def hook(module, input, output):
            nonlocal target_init
            output = output.clone()  # ← éviter modification in-place d'une vue sur un leaf variable
            for i, idx in enumerate(lookup_idxs):
                output[i, idx, :] = output[i, idx, :] + delta
            if target_init is None:
                target_init = output[0, lookup_idxs[0]].detach().clone()
            return output

        editor.accroche(hook)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        editor.enleve()
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)

        # CrossEntropy sur rewriting prompts
        loss_ce = F.nll_loss(
            log_probs[:len(rewriting_inputs)].transpose(1, 2),
            rewriting_targets,
            ignore_index=-100
        )

        # KL sur prompts de contrôle
        kl_idxs = lookup_idxs[len(rewriting_inputs):]
        kl_logits = logits[len(rewriting_inputs):][torch.arange(len(kl_prompts)), kl_idxs]
        kl_log_probs = F.log_softmax(kl_logits, dim=-1)
        if kl_distr_init is None:
            kl_distr_init = kl_log_probs.detach()
        kl_loss = F.kl_div(kl_log_probs, kl_distr_init, log_target=True, reduction="batchmean")

        # Régularisation
        wd_loss = weight_decay * (delta.norm() / (target_init.norm() + 1e-6))**2

        # Total loss
        loss = loss_ce + lambda_kl * kl_loss + wd_loss
        loss.backward()
        optimizer.step()

        # Clamp L2
        max_norm = clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta.mul_(max_norm / delta.norm())

        # Logs
        CE_list.append(loss_ce.item())
        KL_list.append(kl_loss.item())
        loss_list.append(loss.item())

        if step % 10 == 0 or loss.item() < early_stop_threshold:
            print(f"[{step}] Total Loss = {loss.item():.6f} | CE = {loss_ce.item():.6f} | KL = {kl_loss.item():.6f}")
        if loss.item() < early_stop_threshold:
            print(f"\nEarly stopping at iteration {step} with loss {loss.item():.6f}")
            break
    editor.enleve()
    target=target_init+delta
    cur_input,cur_output=get_module_input_output_at_word(editor,input_ids[0],attention_mask[0],offset_mapping[0],factual_prompts[0].format(subject=editor.instance.subject),editor.instance.subject)
    kStar = kStar.to(device)
    W_proj = instance.model.transformer.h[instance._l_star].mlp.c_proj.weight.detach()
    k_star_proj = W_proj.T @ kStar  # shape [1600]


    v_star=(target - cur_output)/torch.dot(cur_input, k_star_proj)

    return v_star.detach(), CE_list, KL_list, loss_list



