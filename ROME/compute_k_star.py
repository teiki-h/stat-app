from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import torch
from functools import partial
import torch.nn.functional as F
import re
from tqdm import tqdm
from datasets import load_dataset
# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
class Instance_for_ROME:
    def __init__(self, subject, inputs=None, l_star=18, model_name='gpt2-xl',C=None, nb_prompt=50,batch_size=2):
        
        self.model_name = model_name
        self.subject = subject
        self._l_star = l_star
        self.batch_size = batch_size
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print(f"Model {model_name} loaded on {self.device}")
        if inputs is None:
            self.prompts=self.generate_prompts(nb_prompt,batch_size=batch_size)
            self.nb_prompt = len(self.prompts)
        else:
            self.prompts = inputs
            self.nb_prompt = len(inputs)
        
        self._k_star = None
        self._hooks = []
        self._logits = None
        self.output = None
        self.activationsC=None
        self.C=C

    def __str__(self):
        return f'Instance of {self.model.config.architectures[0]} model'

    def tokenize(self, batch, offsetsMapping=False):
        inputs = self.tokenizer(batch, return_tensors='pt', padding=True, return_offsets_mapping=offsetsMapping)
        return {k: v.to(self.device) for k, v in inputs.items()}

    def compute_subject_mask(self, prompts=None, subject=None):
        if prompts is None:
            prompts = self.prompts
        if subject is None:
            subject = self.subject

        input = self.tokenize(prompts, offsetsMapping=True)
        mask = []
        for j, prompt in enumerate(prompts):
            map = torch.zeros_like(input['input_ids'][j], dtype=torch.int)
            indexSubject = prompt.find(subject)
            for i, t in enumerate(input['offset_mapping'][j]):
                if indexSubject != -1:
                    if (indexSubject <= t[0]) and (t[1] <= indexSubject + len(subject)):
                        map[i] = 1
            mask.append(map)
        subject_mask = torch.stack(mask)
        subject_mask = torch.logical_and(subject_mask, input['attention_mask']).int()
        return subject_mask
    
    def compute_last_subject_indices(self, prompts):
        subject_mask = self.compute_subject_mask(prompts)
        last_subject_indices = (
            subject_mask * torch.arange(1, subject_mask.shape[1] + 1, device=subject_mask.device)
        ).argmax(dim=1)
        return last_subject_indices

    def get_ks_hook(self, prompts):
        last_subject_indices = self.compute_last_subject_indices(prompts)

        def hook(module, input, output):
            res = input[0][torch.arange(len(last_subject_indices)), last_subject_indices]   # We have to read the value right after the non-linearity of the MLP
            if self._k_star is None:
                self._k_star = res.mean(dim=0)
                self._kcount = 1
            else:
                self._k_star = (self._k_star * self._kcount + res.mean(dim=0)) / (self._kcount + 1)
                self._kcount += 1

        return hook

    def accroche(self, hook,l_star=None):
        if l_star is None:
            l_star = self._l_star
        handle = self.model.transformer.h[l_star].mlp.act.register_forward_hook(hook)
        self._hooks.append(handle)

    def enleve(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks = []

    def run(self, prompts,conserve_logits=False, conserve_output=False):
        input = self.tokenize(prompts)
        with torch.no_grad():
            output = self.model(**input, labels=input['input_ids'])
        if conserve_logits:
            self._logits = output.logits
        if conserve_output:
            self._output = output
            
    def get_k_star(self, l_star=None,batch_size=None):
        if l_star is None:
            l_star = self._l_star
        if batch_size is None:
            batch_size = self.batch_size
        print(f'Getting k_star for {self.nb_prompt//batch_size} batches ...')
        for i in tqdm(range(self.nb_prompt//batch_size)):
            self.accroche(self.get_ks_hook(self.prompts[i*batch_size:(i+1)*batch_size]),l_star=l_star)
            self.run(self.prompts[i*batch_size:(i+1)*batch_size])
            self.enleve()
        self._k_star = self._k_star.cpu()
        if self.C is None:
            self.get_C(self.get_wikipedia_data(100),l_star=l_star,batch_size=batch_size)
        #self._k_star = torch.inverse(self.C) @ self._k_star.unsqueeze(1)
        #self._k_star = self._k_star.squeeze()
        #self._k_star = self._k_star / self._k_star.norm()
  
        return self._k_star
    

    def generate_prompts(self, nb_prompt, handPrompts=None,min_len=2, max_len=11,batch_size=None,mode="k*"):
        prompts= []
        if handPrompts is None:
            handPrompts = [""]
        if batch_size is None:
            batch_size = self.batch_size

        print(f'Generating prompts {batch_size} by {batch_size}...')
        for j in tqdm(range(nb_prompt//batch_size)):   #There won't always be nb_prompt generated but it's ok, choose a multiple of batch_size if you want to be sure
            for i in range(batch_size):
                prompt=self.model.generate(input_ids=self.tokenizer.encode("<|endoftext|>", return_tensors="pt").to(self.device),
                                            max_length=max_len+1 , #to account for the end of text token
                                            min_length=min_len,
                                            num_return_sequences=1,
                                            do_sample=True,
                                            pad_token_id=self.tokenizer.eos_token_id,
                )
                decodedPrompt=  self.tokenizer.decode(prompt[0], skip_special_tokens=True)
                if mode == "k*":
                    prompts.append(decodedPrompt+". "+handPrompts[(j*batch_size+i)%len(handPrompts)]+self.subject)
                elif mode == "v*":
                    prompts.append(decodedPrompt+". "+handPrompts[(j*batch_size+i)%len(handPrompts)].format(subject=self.subject))
                else:
                    print("Error: mode not recognized")
        return prompts
    
    #Calculating the C matrix

    def get_C_hook(self, attentionMask):
        mask=attentionMask.bool()
        def hook(module, input, output):
            activations= output[mask].view(-1,output.size(-1)).cpu()
            self.activationsC.append(activations)
        return hook

    def get_C(self, texts,l_star=None,batch_size=2):
        print(f'Computing C')
        self.activationsC = []
        if l_star is None:
            l_star = self._l_star
        for i in tqdm(range(len(texts)//batch_size)):
            batch= texts[i*batch_size:(i+1)*batch_size]
            input= self.tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
            input_ids = input['input_ids'].to(self.device)
            attention_mask = input['attention_mask'].to(self.device)
            hook=self.get_C_hook(attention_mask)
            self.accroche(hook,l_star=l_star)
            with torch.no_grad():
                # Forward pass on the model (no gradients needed)
                self.model(input_ids=input_ids, attention_mask=attention_mask)
            self.enleve()
            del input_ids
            del attention_mask
            torch.cuda.empty_cache()
        # Compute the kkT_matrices and C
        self.activationsC = torch.cat(self.activationsC, dim=0)
        self.C = self.activationsC.T @ self.activationsC/ self.activationsC.size(0)
        self.C = self.C
        return self.C

    def get_wikipedia_data(self, n):
        ds_name = 'wikitext'

        raw_ds = load_dataset(ds_name, dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")[ds_name])
        def clean_text(text_data):
            cleaned_text_data = []
            for line in text_data:

                line = line.replace('@-@', '-')
                line = line.replace(' @,@ ', ',')
                line = line.replace(' @.@ ', '.')
                line = re.sub(r'\s+', ' ', line).strip()
                line = line.replace("\\'", "'") # ne marche pas je veux remplacer les \' par ' mais j'y arrive pas
                
                # 3. Avoid adding empty lines
                if line:  # Only add non-empty lines
                    cleaned_text_data.append(line)
            cleaned_text_data = [ line for line in cleaned_text_data 
                                    if not (line.startswith('=') and line.endswith('='))
            ]
            return cleaned_text_data
        text_data = raw_ds['train'].shuffle()['text'][:n]
        return clean_text(text_data)

    def delete_instance(self):
        self.model = None
        self.tokenizer = None
        self._k_star = None
        self._hooks = []
        self._logits = None
        self.output = None
        self.activationsC = None
        self.C = None
        torch.cuda.empty_cache()
        print("Instance deleted and GPU memory cleared.")