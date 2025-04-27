
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import torch
from functools import partial


class Instance :
    def __init__(self,model_name,inputs,device='cpu',batch_size=50):
        self.device=torch.device('cpu')

        #Setup device GPU if asked and available, else CPU
        if device=='cuda' or device=='gpu':
            if torch.cuda.is_available():
                self.device=torch.device('cuda')
            else:
                print('GPU not available, running on CPU')

        self.model=GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.tokenizer=GPT2TokenizerFast.from_pretrained(model_name)
        self.tokenizer.pad_token =  self.tokenizer.eos_token

        self.inputs=inputs
        self.batch_size=batch_size
        self.prompts=[dict['prompt'] for dict in inputs]
        self.subjects=[dict['subject'] for dict in inputs]
        self.attributes=[" "+dict['attribute'] for dict in inputs]      #ajout d'un espace pour éviter les problèmes de tokenisation et se ramener à la sitation de la prompt (il y a un espace avant le mot à prédire)

        ## Create batches of prompts, subjects and attributes to ensure that each batch fits in the GPU memory
        self.batchedPrompts=[self.prompts[i:i+self.batch_size] for i in range(0,len(self.prompts),self.batch_size)]
        self.batchedSubjects=[self.subjects[i:i+self.batch_size] for i in range(0,len(self.subjects),self.batch_size)]
        self.batchedAttributes=[self.attributes[i:i+self.batch_size] for i in range(0,len(self.attributes),self.batch_size)]

        self.hidden_states=[]
        self.clean_logits=[]

    def __str__(self):
        return f'Instance of {self.model.config.architectures[0]} model'
    
    def tokenize(self,batch,offsetsMapping=False):
        inputs=self.tokenizer(batch,return_tensors='pt',padding=True,return_offsets_mapping=offsetsMapping)
        inputs.to(self.device)
        return inputs
    
    def subject_mask(self,inputTokens,batchNumber):
        """récupère un tenseur inputTokens correspondant à un batch ainsi que le numéro dudit batch, et retourne un tenseur de la même taille que inputTokens, mais avec 1 pour les tokens qui correspondent au sujet, et 0 sinon
        """
        prompts=self.batchedPrompts[batchNumber]
        subjects=self.batchedSubjects[batchNumber]
        mask = []
        for j, prompt in enumerate(prompts):
            map = torch.zeros_like(input.input_ids[j], dtype=torch.int)#input_ids = id du token, fait un tenseur de 0 de la même dimension
            for i,t in enumerate(input.offset_mapping[j]):#offset_mapping = où est-ce qu'on a mis le padding, i = position, 
                
                if (prompt.find(subjects[j])-1<=t[0]) and (t[1]<=prompt.find(subjects[j])+len(subjects[j])):#sélectionne aussi le padding, qu'on élimine avec logical
                    map[i] = 1
            mask.append(map)
        subjectMask = torch.stack(mask)
        subjectMask = torch.logical_and(subjectMask, input.attention_mask).int()
        for i in inputTokens:
            i.drop('offset_mapping')
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
        for batch in self.batchedPrompts:
            input=self.tokenize(batch)
            with torch.no_grad():
                outputs=self.model(**input,labels=input.input_ids,output_hidden_states = True, output_attentions= True)
                self.clean_logits.append(self.last_non_padding_token_logits(outputs.logits,input.attention_mask))
                self.hidden_states.append(outputs.hidden_states)
            del input,outputs
            torch.cuda.empty_cache()
        return self.clean_logits,self.hidden_states
    
    def proba_clean(self):
        proba_clean=[]
        if self.clean_logits==[]:
            self.clean_run()
        for logits,correctWords in zip(self.clean_logits,self.batchedAttributes):
            for logit,correctWord in zip(logits,correctWords):
                correctId=self.tokenizer(correctWord,add_special_tokens=False)['input_ids']
                correctId=correctId[0]
                proba=torch.nn.functional.softmax(logit,dim=-1)[correctId].item()
                proba_clean.append(proba)
        return proba_clean
    

    def create_noise_hook(self,inputTokens,batchNumber):
        subjectMask=self.subject_mask(inputTokens,batchNumber)
        def noise_hook(module, input,output):
            std_dev_all = torch.std(output.flatten())
            noise = torch.randn_like(output)*3*std_dev_all
            noisy_output = output + noise * subjectMask.unsqueeze(-1).float()
            return noisy_output
        return noise_hook
    
    def corrupted_run(self):
        for i,batch in enumerate(self.batchedPrompts):
            input=self.tokenize(batch)
            self.model.transformer.wpe.register_forward_hook(self.create_noise_hook(input,i))
            with torch.no_grad():
                outputs=self.model(**input,labels=input.input_ids,output_hidden_states = True, output_attentions= True)
                self.clean_logits.append(self.last_non_padding_token_logits(outputs.logits,input.attention_mask))
                self.hidden_states.append(outputs.hidden_states)
            del input,outputs
            torch.cuda.empty_cache()
        return self.clean_logits,self.hidden_states