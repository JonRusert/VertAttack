import sys
import csv
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification


class HuggingFaceModel:
    
    def __init__(self, hfmodel = 'textattack/bert-base-uncased-imdb'):
        self.tokenizer = AutoTokenizer.from_pretrained(hfmodel)
        self.model = AutoModelForSequenceClassification.from_pretrained(hfmodel)


    def predict(self, test_query, attacked_class = 0):
        #print(test_query)
        
        # Default max length is set to be int(1e30), so we force 512 to enable batching.
        max_length = (
            512
            if self.tokenizer.model_max_length == int(1e30)
            else self.tokenizer.model_max_length
        )
        
        text_input_list = [test_query]

        inputs_dict = self.tokenizer(
            text_input_list,
            add_special_tokens=True,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        

        #tok_text = self.tokenizer([test_query], return_tensors='pt')
        #print(tok_text)
        #output = self.model(**tok_text)[0]
        output = self.model(**inputs_dict)[0]

        #print(output)
        softmax = torch.nn.Softmax(dim=1)
        output_sm = softmax(output)[0]
        #print(output_sm)
            
        pred = float(output_sm.argmax())
        
        prob = output_sm[int(attacked_class)].detach().numpy()
        
        return pred, float(prob)



    # allow multiple queries in form of list
    def predictMultiple(self, test_queries, attacked_class = 0):
        predictions = []
        probs = []
        for test in test_queries:
            pred, prob = self.predict(test, attacked_class)
            predictions.append(pred)
            probs.append(prob)

        return predictions, probs


        
        

'''
text= 'this movie was very good 10/10'

tokenizer = AutoTokenizer.from_pretrained('textattack/bert-base-uncased-imdb')
model = AutoModelForSequenceClassification.from_pretrained('textattack/bert-base-uncased-imdb')

tok_text = tokenizer([text], return_tensors='pt')
print(tok_text)
output = model(**tok_text)[0]
print(output)
softmax = torch.nn.Softmax(dim=1)
output_sm = softmax(output)[0]

pred = float(output_sm.argmax())
        
        
print(output_sm)
print(pred)


hfm = HuggingFaceModel()
pos_text = 'this movie was very good 10/10'
neg_text = 'this movie was horrible 0/10'
print(hfm.predict(pos_text, 1))
print(hfm.predict(neg_text, 0))
'''

