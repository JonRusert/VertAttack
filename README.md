# VertAttack
Contains code and generated results for VertAttack: Taking advantage of Text Classifiers’ horizontal vision
Temp arxiv url: https://arxiv.org/abs/2404.08538


# Example Usage:

vertattacker = VertAttack(obf_method = 'GS_HRP', class_alg = 'HuggingFaceModel', hfmodel = 'textattack/distilbert-base-uncased-imdb', p_chaff = 0.3)                       
out = vertattacker.obfuscate('one of the worst movies of the year . . . watching it was painful')                                                                          
out = vertattacker.obfuscate('worst movie ever')                                                                                                                           
print(out[0], out[1])            
