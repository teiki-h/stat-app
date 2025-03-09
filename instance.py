
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import torch

class Instance :
    def __init__(self,model_name,inputs,engine='cpu',batch_size=50):
        self.device=torch.device('cpu')
        if engine=='cuda' or engine=='gpu':
            if torch.cuda.is_available():
                self.device=torch.device('cpu')
            else:
                print('GPU not available, running on CPU')

        self.model=GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.tokenizer=GPT2TokenizerFast.from_pretrained(model_name)
        self.tokenizer.pad_token =  self.tokenizer.eos_token
        self.inputs=inputs
        self.batch_size=batch_size
        self.prompts=[dict['prompt'] for dict in inputs]
        self.subjects=[dict['subject'] for dict in inputs]
        self.inputTokens=self.tokenize()
        self.subjectMask=self.subject_mask()

    def __str__(self):
        return f'Instance of {self.model.config.architectures[0]} model'
    
    def tokenize(self):
        batchedPrompts=[self.prompts[i:i+self.batch_size] for i in range(0,len(self.prompts),self.batch_size)]
        inputTokens=[]
        for batch in batchedPrompts:
            inputs=self.tokenizer(batch,return_tensors='pt',padding=True,return_offsets_mapping=True)
            inputs.to(self.device)
            inputTokens.append(inputs)
        return inputTokens
    
    def subject_mask(self):
        batchedPrompts=[self.prompts[i:i+self.batch_size] for i in range(0,len(self.prompts),self.batch_size)]
        batchedSubjects=[self.subjects[i:i+self.batch_size] for i in range(0,len(self.subjects),self.batch_size)]
        subjectMask=[]
        for prompts,subjects,input in zip(batchedPrompts,batchedSubjects,self.inputTokens):
            mask = []
            for j, prompt in enumerate(prompts):
                map = torch.zeros_like(input.input_ids[j], dtype=torch.int)#input_ids = id du token, fait un tenseur de 0 de la même dimension
                for i,t in enumerate(input.offset_mapping[j]):#offset_mapping = où est-ce qu'on a mis le padding, i = position, 
                    
                    if (prompt.find(subjects[j])-1<=t[0]) and (t[1]<=prompt.find(subjects[j])+len(subjects[j])):#sélectionne aussi le padding, qu'on élimine avec logical
                        map[i] = 1
                mask.append(map)
            masks_tensor = torch.stack(mask)
            masks_tensor = torch.logical_and(masks_tensor, input.attention_mask).int()
            subjectMask.append(masks_tensor)
        return subjectMask
    
    def last_non_padding_token_logits(self,logits,attention_mask):
        """récupère un tenseur logits, et attention mask, et retourne un tenseur donnant, pour chauque logit de chaque prompt, le logit du dernier mots
        """
        # For each input, find the last non-padding token
        last_non_padding_logits = []
        
        for i in range(logits.size(0)):  # Loop over each prompt in the batch
            # Find the last non-padding token position
            non_padding_positions = (attention_mask[i] == 1).nonzero(as_tuple=True)[0]
            last_non_padding_token_index = non_padding_positions[-1]
            
            # Get the logits of the last non-padding token
            last_non_padding_logits.append(logits[i, last_non_padding_token_index])
        last_non_padding_logits = torch.stack(last_non_padding_logits)
        return last_non_padding_logits
    
    def clean_run(self):
        