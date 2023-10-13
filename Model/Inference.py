from transformers import AutoModelWithLMHead, AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM
import pandas as pd
from pathlib import Path

def loadChatContext(filepath, num_blocks_per_session):
    df = pd.read_csv(str(filepath), encoding='utf-8')
    sess_idx_mssg = 0
    history = ''
    for i, data in df.iterrows():
        if data[1] != '#':
            if sess_idx_mssg < num_blocks_per_session:
                history = history + '<user>:' + data[0] + '| <bot>:' + data[1] + '|\n'
                sess_idx_mssg += 1
            else:
                break
    return history


def answer(input_text, model, tokenizer):
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    generated_ids = model.generate(input_ids=input_ids.to('cuda'),
                                   do_sample=True,
                                   max_new_tokens=10,
                                   num_beams = 20,
                                   top_p=0.9,
                                   top_k=20,
                                   no_repeat_ngram_size=2,
                                   pad_token_id=tokenizer.pad_token_id)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    # Process or use the generated text as needed
    return generated_text

def chat(role, nonrole, model, tokenizer, context):

    for i in range(1, 100):
        query = input('<user>:')
        if i==1:
            query = f"{role}:{query}\n{nonrole}:"
            query = context + query
        if i>1:
            query = response + f"{role}:{query}\n{nonrole}:"
        response = " ".join(answer(query, model, tokenizer).split("| ")) + '\n|\n'
        print(response)

model_dir = './Scratch/last_model'
tokenizer_dir = './Scratch/tokenizer/'
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)
model.to('cuda')

chat_path = Path('./PreparedDataV3/Chat de WhatsApp con Marti.csv')
chat_context = loadChatContext(chat_path, 2)

chat('<user>', '<bot>', model, tokenizer, '')