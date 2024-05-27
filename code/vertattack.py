

import sys
sys.path.append('Classifiers/GooglePerspective/')
sys.path.append('Classifiers/HuggingFaceModels/')
from Perspective import Perspective
from huggingfacemodel import HuggingFaceModel
import csv
import wordsegment
import random

class VertAttack:


    def __init__(self, obf_method = 'GS_RP', class_alg = 'Perspective', persp_key = 1, hfmodel = 'textattack/bert-base-uncased-imdb', p_chaff = 0.0):
        self.obf_method = obf_method
        self.class_alg = class_alg
        if(class_alg == 'Perspective'):
            self.classifier = Perspective(threshold = 0.5, select = persp_key)
        elif(class_alg == 'HuggingFaceModel'):
            self.classifier = HuggingFaceModel(hfmodel)

        self.p_chaff = p_chaff
        self.query_count = 0

    # takes in list of words and list of indices where the word is to be perturbed
    # follows a simple rule based method to perturb the texts vertically. 
    def rulePerturb(self, words, perturb_indices):
        if(len(perturb_indices) == 0):
            num_lines = 1
        else:
            num_lines = max([len(words[i]) for i in perturb_indices])

        perturb_text = ''
        cur_line = 0 # keeps track of how deep the text is

        while(cur_line < num_lines): 
            for i in range(len(words)):

                # if word is to be perturbed, insert letter character at a time
                if(i in perturb_indices):
                    if(cur_line < len(words[i])):
                        perturb_text += words[i][cur_line] + ' '
                    else: 
                        perturb_text += '  '
                else:
                    if(cur_line == 0):
                        perturb_text += words[i] + ' '
                    else: # need to pad the space
                        #perturb_text += len(words[i]) * ' ' + ' '
                        for k in range(len(words[i])):
                            # add chaff with prob p_chaff, if p_chaff = 0.0 then space only added
                            # also do not add chaff if right next to actual perturbed text
                            if(random.random() < self.p_chaff and (i+1) not in perturb_indices):
                                rand_char = random.choice('abcdefghijklmnopqrstuvwxyz')
                                perturb_text += rand_char
                            else:
                                perturb_text += ' '
                            
                        perturb_text += ' ' 

                            


            perturb_text += '\n'
            cur_line += 1

        perturb_text = perturb_text[:-1] # remove last newline 


        return perturb_text




    # restricts perturbed text to a specific word width 
    def hRestRulePerturb(self, words, perturb_indices, width = 5):
        cur_width = 0
        done = False
        final_text = ''
        while(not done):
            i = 0
            tmp_words = []

            # process words width at a time using base method
            while((i < width) and (len(words) > 0)):
                tmp_words.append(words.pop(0))
                i += 1



            tmp_indices = []
            for j in perturb_indices:
                adj_j = j - cur_width
                if(0 <= adj_j < width):
                    tmp_indices.append(adj_j)

            cur_text = self.rulePerturb(tmp_words, tmp_indices)

            if(cur_width == 0):
                final_text = cur_text
            else:
                final_text += '\n' + cur_text 

            cur_width += width
            if(len(words) == 0):
                done = True


        return final_text


    # beg sentence is there for tasks like nli which take 2 sentences (beg, query) and both are needed to determine which word to drop in query
    def GreedySelect(self, query, attack_class = 0, beg_sentence = None):
        orig_text = query
        #query = self.preProcessText(query)

        # get initial probability for query 
        if(beg_sentence):
            _, initial_prob = self.classifier.predict((beg_sentence, query), attack_class)
            self.query_count += 1
        
        else:
            _, initial_prob = self.classifier.predict(query, attack_class)
            self.query_count += 1
        

        #print(query, initial_prob)
        needsReplacing = []
        split_query = query.split()
            
        variations = []
        prob_diffs = []

        # step through each word and generate the variations of the original query by removing one word at a time
        for cur_pos in range(len(split_query)):
            modified_query = ' '.join(split_query[:cur_pos] + split_query[cur_pos+1:])
            
            if(beg_sentence): #nli tasks, attack second sentence, but need first for classification
                variations.append((beg_sentence, modified_query))
            else:
                variations.append(modified_query)

        # get probabilities for all variations

        orig_preds, var_probs = self.classifier.predictMultiple(variations, attack_class)
        self.query_count += len(variations)
        

        for cur_prob in var_probs:
            prob_diffs.append(initial_prob - cur_prob)
        

        replace_pos = prob_diffs.index(max(prob_diffs))
        
        # not currently used
        replace_pred = orig_preds[replace_pos]
        replace_prob = var_probs[replace_pos]

        return replace_pos

    

        

    

    # GreedySelect HorizontalRulePeturb
    def GS_HRP(self, query, attack_class = 0, width = 10, preprocess = None):
        self.query_count = 0

        first_sentence = None        
        if(type(query) == tuple): # for nli tasks, only attack the response sentence
            first_sentence = query[0]
            query = query[1]

        perturbed_query = query
        orig_pos = {}    
        split_query = query.split()
            
        #note original position so when multiple are removed, original is retained
        for i in range(len(split_query)):
            orig_pos[i] = i

        needsReplacing = []
        done = False

        while(not done):
            # while
            ## Select Word to replace, replace, then check if tricked classifier, repeat until this does so
            next_replace = self.GreedySelect(' '.join(split_query), attack_class, first_sentence)

            orig_repl = orig_pos[next_replace]

            needsReplacing.append(orig_repl)
            split_query.pop(next_replace)

            ## Perturb Step
            # Perturb all the candidate words vertically, limit the width
            perturbed_query = self.hRestRulePerturb(query.split(), needsReplacing, width)

            
            # preprocess like classifier would if present
            if(preprocess):
                if(preprocess == 'simple'):
                    processed_query = ' '.join(perturbed_query.split())
                elif(preprocess == 'segmenter'):
                    processed_query = ' '.join(wordsegment.segment(perturbed_query))


            #check if perturbed fools classifier
            if(first_sentence): # for nli tasks, add back in first sentence
                if(preprocess):
                    processed_query = (first_sentence, processed_query)
                
                perturbed_query = (first_sentence, perturbed_query)

            if(preprocess):
                pert_pred, pert_prob = self.classifier.predict(processed_query, attack_class)
            else:
                pert_pred, pert_prob = self.classifier.predict(perturbed_query, attack_class)

            self.query_count += 1
        
            #print(pert_pred, pert_prob, '\n', perturbed_query)

            if(pert_pred != attack_class or len(split_query) == 0):
                done = True
            else: # update original positions
                for i in range(len(split_query)):            
                    if(i < next_replace):
                        orig_pos[i] = orig_pos[i]
                    elif(i >= next_replace):
                        orig_pos[i] = orig_pos[i+1]




        
        return perturbed_query, self.query_count






    def obfuscate(self, query, attack_class = 0, width = 10, preprocess = None):
        if(self.obf_method == 'GS_HRP'):
            return self.GS_HRP(query, attack_class, width, preprocess)



def main(dataset = 'imdb.tsv', num_examples = 1000, offset = 0, outfile = 'output.tsv', classifier = 'HuggingFaceModel', hfmodel = 'textattack/bert-base-uncased-imdb', width = 10, preprocess = None, p_chaff = 0.0):
    obf_method = "GS_HRP"
    p_chaff = float(p_chaff)
    vertattacker = VertAttack(obf_method = obf_method, class_alg = classifier, hfmodel = hfmodel, p_chaff = p_chaff)
    if(preprocess == "None"):
        preprocess = None
    

    incsv = csv.reader(open(dataset), delimiter = '\t')
    outcsv = csv.writer(open(outfile, 'w'), delimiter = '\t')
    width = int(width)


    if(preprocess == 'segmenter'):
        wordsegment.load()


    num_examples = int(num_examples)
    offset = int(offset)
    if(num_examples == -1):
        num_examples = len(incsv)

    count = 0
    cur_num = 0
    for cur in incsv:
        #head to offset of dataset
        if(not (offset <= cur_num)):
            cur_num += 1
            continue

        if(len(cur) == 2):
            ground_truth = int(cur[1])
            text = cur[0]
        elif(len(cur) == 3): # catches datasets which have idx in first column (e.g. sst2)
            ground_truth = int(cur[2])
            text = cur[1]
        else: # for nli tasks with multiple sentences (assumes idx as well), vertattack only attacks the second text. 
            ground_truth = int(cur[3])
            text = (cur[1], cur[2])

        perturbed_text, num_queries = vertattacker.obfuscate(text, ground_truth, width, preprocess)

        if(type(perturbed_text) == tuple): # for nli tasks
            outcsv.writerow([perturbed_text[0], perturbed_text[1], ground_truth, num_queries])
        else:
            outcsv.writerow([perturbed_text, ground_truth, num_queries])

        count += 1
        if(count >= num_examples):
            break




# Example

#vertattacker = VertAttack(obf_method = 'GS_HRP', class_alg = 'HuggingFaceModel', hfmodel = 'textattack/distilbert-base-uncased-imdb', p_chaff = 0.3)
#out = vertattacker.obfuscate('one of the worst movies of the year . . . watching it was painful')
#out = vertattacker.obfuscate('worst movie ever')
#print(out[0], out[1])

#print(vertattacker.rulePerturb('they are very sad people, they are lost and broken'.split(), [3, 7, 9]))
#print(vertattacker.hRestRulePerturb('they are very sad people, they are lost and broken'.split(), [3, 7, 9], 5))


if(__name__ == "__main__"):
    if(len(sys.argv) == 10):
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8], sys.argv[9])
    elif(len(sys.argv) == 9):
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])
    elif(len(sys.argv) == 8):
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
    else:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])


