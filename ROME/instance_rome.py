from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import torch
from functools import partial

class Instance_ROME :
    def __init__(self, subject, inputs= None, l_star = 18, model_name = 'gpt2-xl', device="cpu", batch_size=50,nb_prompt=50):
        self.model_name = model_name
        self.subject = subject
        self._l_star = l_star

        #Setup device GPU if asked and available, else CPU
        self.device=torch.device('cpu')
        if device=='cuda' or device=='gpu':
            if torch.cuda.is_available():
                self.device=torch.device('cuda')
            else:
                print('GPU not available, running on CPU')


        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if inputs == None:
            self.generate_prompts(50)
        else:
            self.prompts = inputs

        self._subject_mask = self.compute_subject_mask()
        self._last_subject_indices= (self._subject_mask * torch.arange(1, self._subject_mask.shape[1] + 1, device=self._subject_mask.device)).argmax(dim=1)

        self._ks = None
        self._k_star=None
        self._hooks = []
        self._logits = None
        self.output = None

    def __str__(self):
        return f'Instance of {self.model.config.architectures[0]} model'
    
    def tokenize(self,batch,offsetsMapping=False):
        inputs=self.tokenizer(batch,return_tensors='pt',padding=True,return_offsets_mapping=offsetsMapping)
        inputs.to(self.device)
        return inputs
    
    def compute_subject_mask(self, prompts = None, subject = None):
        res =[]

        if prompts == None:
            prompts = self.prompts
        if subject == None:
            subject = self.subject

        input = self.tokenize(prompts,offsetsMapping=True)
        mask=[]
        for j, prompt in enumerate(prompts):
            map = torch.zeros_like(input.input_ids[j], dtype=torch.int)
            for i,t in enumerate(input.offset_mapping[j]):
                if (prompts[j].find(subject)-1<=t[0]) and (t[1]<=prompts[j].find(subject)+len(subject)) and (prompts[j].find(subject) !=-1):
                    map[i] = 1
            mask.append(map)
        subject_mask = torch.stack(mask)
        subject_mask = torch.logical_and(subject_mask, input.attention_mask).int()
        return subject_mask
    
    def get_ks_hook(self, last_subject_indices = None):
        if last_subject_indices == None:
            last_subject_indices = self._last_subject_indices
        
        def hook(module,input,output):
            if isinstance(output, torch.Tensor):
                res = output[torch.arange(len(last_subject_indices)), last_subject_indices]
                self._ks = res
            else:
                raise TypeError("Expected output to be a torch.Tensor, but got {}".format(type(output)))
            pass
        
        return hook
    
    def accroche(self, l_star = None):
        if l_star == None:
            l_star = self._l_star
        hook = self.get_ks_hook()
        handle = self.model.transformer.h[l_star].mlp.c_fc.register_forward_hook(hook)
        self._hooks.append(handle)
        pass
    
    def enleve(self):
        for handle in self._hooks:
            handle.remove()
        self._hooks = []
        pass
    
    def run(self, conserve_logits = False,conserve_output = False):
        input = self.tokenize(self.prompts)
        with torch.no_grad():
            output = self.model(**input, labels = input.input_ids) 
        if self._ks != None:
            self._k_star = torch.mean(self._ks, dim=0)
        if conserve_logits:
            self._logits = output.logits 
        if conserve_output:
            self._output = output
        pass

    def generate_prompts(self, nb_prompt, min_len = 2, max_len = 11):
        vocab_size = self.tokenizer.vocab_size
        nb_token = torch.randint(min_len, max_len, (nb_prompt,))
        max_tokens = nb_token.max() 
        tokens = torch.randint(0, vocab_size, (nb_prompt, max_tokens))
        padded_tokens = F.pad(tokens, (0, max_tokens - nb_token.max().item()), value=vocab_size)
        decoded_sequences = [self.tokenizer.decode(seq[:nb_token[i].item()]) for i, seq in enumerate(padded_tokens)]
        res = [x + ' ' + self.subject for x in decoded_sequences]
        self.__init__(self.subject, res, self._l_star,self.model_name)
        pass

    def get_k_star(self,l_star = None):
        self.accroche(l_star)
        self.run()
        self.enleve()
        return self._k_star