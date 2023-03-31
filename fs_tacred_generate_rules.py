"""
Generate rules for Few-Shot TACRED evaluation
"""

import json
import tqdm
import torch

from transformers import RobertaTokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import Dataset

device = torch.device('cuda:1')

model     = T5ForConditionalGeneration.from_pretrained('output1/checkpoint-100000').to(device)
tokenizer = RobertaTokenizer.from_pretrained('output1/checkpoint-100000')


paths = [
    '/data/nlp/corpora/softrules/tacred_fewshot/train/5_way_1_shots_10K_episodes_3q_seed_160290.json',
    '/data/nlp/corpora/softrules/tacred_fewshot/train/5_way_1_shots_10K_episodes_3q_seed_160291.json',
    '/data/nlp/corpora/softrules/tacred_fewshot/train/5_way_1_shots_10K_episodes_3q_seed_160292.json',
    '/data/nlp/corpora/softrules/tacred_fewshot/train/5_way_1_shots_10K_episodes_3q_seed_160293.json',
    '/data/nlp/corpora/softrules/tacred_fewshot/train/5_way_1_shots_10K_episodes_3q_seed_160294.json',
    '/data/nlp/corpora/softrules/tacred_fewshot/train/5_way_5_shots_10K_episodes_3q_seed_160290.json',
    '/data/nlp/corpora/softrules/tacred_fewshot/train/5_way_5_shots_10K_episodes_3q_seed_160291.json',
    '/data/nlp/corpora/softrules/tacred_fewshot/train/5_way_5_shots_10K_episodes_3q_seed_160292.json',
    '/data/nlp/corpora/softrules/tacred_fewshot/train/5_way_5_shots_10K_episodes_3q_seed_160293.json',
    '/data/nlp/corpora/softrules/tacred_fewshot/train/5_way_5_shots_10K_episodes_3q_seed_160294.json',
    '/data/nlp/corpora/softrules/tacred_fewshot/dev/5_way_1_shots_10K_episodes_3q_seed_160290.json',
    '/data/nlp/corpora/softrules/tacred_fewshot/dev/5_way_1_shots_10K_episodes_3q_seed_160291.json',
    '/data/nlp/corpora/softrules/tacred_fewshot/dev/5_way_1_shots_10K_episodes_3q_seed_160292.json',
    '/data/nlp/corpora/softrules/tacred_fewshot/dev/5_way_1_shots_10K_episodes_3q_seed_160293.json',
    '/data/nlp/corpora/softrules/tacred_fewshot/dev/5_way_1_shots_10K_episodes_3q_seed_160294.json',
    '/data/nlp/corpora/softrules/tacred_fewshot/dev/5_way_5_shots_10K_episodes_3q_seed_160290.json',
    '/data/nlp/corpora/softrules/tacred_fewshot/dev/5_way_5_shots_10K_episodes_3q_seed_160291.json',
    '/data/nlp/corpora/softrules/tacred_fewshot/dev/5_way_5_shots_10K_episodes_3q_seed_160292.json',
    '/data/nlp/corpora/softrules/tacred_fewshot/dev/5_way_5_shots_10K_episodes_3q_seed_160293.json',
    '/data/nlp/corpora/softrules/tacred_fewshot/dev/5_way_5_shots_10K_episodes_3q_seed_160294.json',
    '/data/nlp/corpora/softrules/tacred_fewshot/test/5_way_1_shots_10K_episodes_3q_seed_160290.json',
    '/data/nlp/corpora/softrules/tacred_fewshot/test/5_way_1_shots_10K_episodes_3q_seed_160291.json',
    '/data/nlp/corpora/softrules/tacred_fewshot/test/5_way_1_shots_10K_episodes_3q_seed_160292.json',
    '/data/nlp/corpora/softrules/tacred_fewshot/test/5_way_1_shots_10K_episodes_3q_seed_160293.json',
    '/data/nlp/corpora/softrules/tacred_fewshot/test/5_way_1_shots_10K_episodes_3q_seed_160294.json',
    '/data/nlp/corpora/softrules/tacred_fewshot/test/5_way_5_shots_10K_episodes_3q_seed_160290.json',
    '/data/nlp/corpora/softrules/tacred_fewshot/test/5_way_5_shots_10K_episodes_3q_seed_160291.json',
    '/data/nlp/corpora/softrules/tacred_fewshot/test/5_way_5_shots_10K_episodes_3q_seed_160292.json',
    '/data/nlp/corpora/softrules/tacred_fewshot/test/5_way_5_shots_10K_episodes_3q_seed_160293.json',
    '/data/nlp/corpora/softrules/tacred_fewshot/test/5_way_5_shots_10K_episodes_3q_seed_160294.json',
]

def read_all_paths(paths):
    output_data = {}
    for path in paths:
        with open(path) as fin:
            data = json.load(fin)
            for episode, relations in tqdm.tqdm(zip(data[0], data[2]), total=len(data[0])):
                meta_train = episode['meta_train']
                meta_test  = episode['meta_test']
                for support_sentences_per_relation, relation in zip(meta_train, relations[0]):
                    for ss in support_sentences_per_relation:
                        if ss['id'] in output_data:
                            assert(output_data[ss['id']] == ss)
                        else:
                            output_data[ss['id']] = ss
                for test_sentence, relation in zip(meta_test, relations[1]):
                    if test_sentence['id'] in output_data:
                        if output_data[test_sentence['id']] != test_sentence:
                            print("\n")
                            print(output_data[test_sentence['id']])
                            print(test_sentence)
                            print("\n")
                        assert(output_data[test_sentence['id']] == test_sentence)
                    else:
                        output_data[test_sentence['id']] = test_sentence

    return output_data

def convert_line(line):
    if line['subj_end'] < line['obj_start']:
        subj_then_obj_order = True
        first_entity_start  = line['subj_start']
        first_entity_end    = line['subj_end'] + 1
        second_entity_start = line['obj_start']
        second_entity_end   = line['obj_end'] + 1
        first_entity_type   = line['subj_type']
        second_entity_type  = line['obj_type']
    else:
        subj_then_obj_order = False
        first_entity_start  = line['obj_start']
        first_entity_end    = line['obj_end'] + 1
        second_entity_start = line['subj_start']
        second_entity_end   = line['subj_end'] + 1
        first_entity_type   = line['obj_type']
        second_entity_type  = line['subj_type']

    until_first_entity  = line['token'][:first_entity_start]
    first_entity        = line['token'][first_entity_start:first_entity_end]
    inbetween_entities  = line['token'][first_entity_end:second_entity_start]
    second_entity       = line['token'][second_entity_start:second_entity_end]
    after_second_entity = line['token'][second_entity_end:]

    after_first_entity  = line['token'][first_entity_end:]
    until_second_entity = line['token'][:second_entity_start]

    tokens_with_entities = until_first_entity + \
        [f'[{first_entity_type}]'] + first_entity + [f'[{first_entity_type}]'] + \
        inbetween_entities + \
        [f'[{second_entity_type}]'] + second_entity + [f'[{second_entity_type}]'] + \
        after_second_entity
    
    return {
        **line,
        'tokens_with_entities': ' '.join(tokens_with_entities),
        'subj_then_obj_order' : subj_then_obj_order,
        'first_entity_type'   : first_entity_type,
        'second_entity_type'  : second_entity_type,
    }

def generate_rule_for_tokens(tokens, model, tokenizer, device):
    generated = tokenizer.batch_decode(model.generate(**tokenizer(tokens, truncation=True, padding=True, max_length=512, return_tensors='pt').to(device), max_length=192, do_sample=True, top_p=0.95, num_return_sequences=10), skip_special_tokens=True)
    for i in range(len(tokens)):
        current_generated = generated[(i*10):((i+1)*10)]
        current_generated = sorted(list(set(current_generated)))

    return generated

output_data = list(read_all_paths(paths).items())
output_data = [(x[0], convert_line(x[1])) for x in output_data]

rules = []

# data = Dataset.from_list([{'tokens': x} for x in all_tokens]).map(lambda x: {'rule': generate_rule_for_tokens(x['tokens'], model, tokenizer, device)}, batched=True, batch_size=256)
data     = Dataset.from_list([{'tokens': x[1]['tokens_with_entities']} for x in output_data]).map(lambda x: {**tokenizer(x['tokens'], truncation=True, max_length=512)}, batched=True).remove_columns(['tokens'])#.select(range(5000))
collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, max_length=512)
dl       = torch.utils.data.DataLoader(data, batch_size=8, shuffle=False, collate_fn=collator, num_workers=16)

rules = []
for batch in tqdm.tqdm(dl):
    num_return_sequences = 10
    generated = tokenizer.batch_decode(model.generate(**{k:v.to(device) for (k, v) in batch.items()}, max_length=192, do_sample=True, top_p=0.95, num_return_sequences=num_return_sequences), skip_special_tokens=True)
    # generated = tokenizer.batch_decode(model.generate(**{k:v.to(device) for (k, v) in batch.items()}, max_length=192), skip_special_tokens=True)
    for i in range(len(batch['input_ids'])):
        current_generated = generated[(i*num_return_sequences):((i+1)*num_return_sequences)]
        rules.append(list(set(current_generated)))

result = []
for (x, y) in zip(output_data, rules):
    result.append({
        **x[1], 
        'rules': y
    })

with open('data/rules_test.jsonl', 'w+') as fout:
    for line in result:
        _=fout.write(json.dumps(line))
        _=fout.write('\n')




