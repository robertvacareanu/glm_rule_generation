"""
Small utility file to load the model and start generating for some simple examples
The idea of this file is to be fully contained, such that the following requirements are satisfied:
- the trained model is loaded
- all the necessary functions to pre-process the input are loaded (e.g. the tokenizer)
- code to generate the output is present
"""

from transformers import RobertaTokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

model     = T5ForConditionalGeneration.from_pretrained('output1/checkpoint-100000')
tokenizer = RobertaTokenizer.from_pretrained('output1/checkpoint-100000')

examples = [
    "[PERSON] Bill Gates [PERSON] founded [ORGANIZATION] Microsoft [ORGANIZATION]", 
    "[PERSON] John [PERSON] , born in New York City , died at age [NUMBER] 95 [NUMBER]", 
    "[ORGANIZATION] Microsoft [ORGANIZATION] acquired [ORGANIZATION] Blizzard [ORGANIZATION] for a record 50 Billion dollars .", 
    "[PERSON] John [PERSON] and [PERSON] Mary [PERSON] are sibblings .", 
    "[PERSON] Bill Gates [PERSON] founded [ORGANIZATION] Microsoft [ORGANIZATION]", 
    "[PERSON] Bill Gates [PERSON] founded [ORGANIZATION] Microsoft [ORGANIZATION]", 
]

for x in examples:
    print("------------")
    print(x)
    print(tokenizer.decode(model.generate(**tokenizer([x], return_tensors='pt'), max_length=64)[0], skip_special_tokens=True))
    print("------------")
    
# tokenizer.decode(model.generate(**tokenizer([x], return_tensors='pt'), max_length=64)[0], skip_special_tokens=True)
# tokenizer.decode(model.generate(**tokenizer([x], return_tensors='pt'), max_length=64)[0], skip_special_tokens=True)
# tokenizer.decode(model.generate(**tokenizer([x], return_tensors='pt'), max_length=64)[0], skip_special_tokens=True)
# tokenizer.decode(model.generate(**tokenizer([x], return_tensors='pt'), max_length=64)[0], skip_special_tokens=True)
# tokenizer.decode(model.generate(**tokenizer([x], return_tensors='pt'), max_length=64)[0], skip_special_tokens=True)
