import datasets
from datasets import Dataset
from transformers import RobertaTokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
model     = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')

dataset = datasets.load_dataset('json', data_files=['/data/nlp/corpora/rules/enhanced_syntax_rules.jsonl'])

data = []
for line in dataset['train']:
    sentence = line['sentence_tokenized']

    head_entity = line['head_entity']
    head_entity = f'[{head_entity}]'
    
    tail_entity = line['tail_entity']
    tail_entity = f'[{tail_entity}]'
    
    output   = line['query']

    sentence = sentence[:line['match_word_start']] + \
        [head_entity] + sentence[line['match_word_start']:(line['match_word_start'] + line['in_between_entities_start'] - 1)] + [head_entity] + \
        sentence[line['in_between_entities_start']:line['in_between_entities_end']] + \
        [tail_entity] + sentence[line['in_between_entities_end']:line['match_word_end']] + [tail_entity] + \
        sentence[line['match_word_end']:]

    data.append({
        'input' : ' '.join(sentence),
        'output': output,
    })

data = Dataset.from_list(data).train_test_split(test_size=0.2).map(lambda x: {**tokenizer(x['input'], truncation=True, max_length=512), 'labels': tokenizer(x['output'], truncation=True, max_length=512)['input_ids']}, batched=True).remove_columns(['input', 'output'])
train_dataset = data['train']
dev_dataset   = data['test']

output_dir = 'output1'

training_args = Seq2SeqTrainingArguments(
    output_dir                  = output_dir,
    fp16                        = False,
    # fp16_backend                = "amp",
    per_device_train_batch_size = 8,
    per_device_eval_batch_size  = 8,
    # eval_accumulation_steps     = 16,
    evaluation_strategy         = "steps",
    eval_steps                  = 5000,      #logging_steps,
    save_steps                  = 10000,
    logging_steps               = 50,
    save_total_limit            = 2,
    max_steps                   = 100000,
    # gradient_accumulation_steps = len(train_datasets) * 2,
    report_to                   = "wandb",
    remove_unused_columns       = False,
    # weight_decay                = 0.001,
    warmup_ratio                = 0.1,
    lr_scheduler_type           = 'linear',
    dataloader_num_workers      = 16,
    learning_rate               = 3e-4,
    # load_best_model_at_end      = True,
    label_names                 = ['labels']
)

label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8,
)

trainer = Seq2SeqTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_dataset,
    eval_dataset    = dev_dataset,
    # eval_dataset    = {'en_ner': en_ner['validation'].select(range(1000)), 'fr_ner': fr_ner['validation'].select(range(1000))},
    tokenizer       = tokenizer,
    data_collator   = data_collator,
    # compute_metrics = compute_metrics
)

trainer.train()
