import random
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
from transformers import AutoModelWithLMHead, AutoTokenizer, TrainingArguments, Trainer, AutoModelForCausalLM, GPTQConfig
from pathlib import Path
import numpy as np
from datasets import Dataset
import transformers
import pandas as pd
import GenerateDataset as GD
import torch
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
import TrainingData_InOutPairs as TDIO
import TrainingData_UniqueConversations as TDUC
import FineTuning as FT
import matplotlib.pyplot as plt


root_dir = Path('PreparedDataV3/')


tokenizer = AutoTokenizer.from_pretrained("datificate/gpt2-small-spanish")
#tokenizer = AutoTokenizer.from_pretrained("ncoop57/DiGPTame-medium")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.add_special_tokens({'additional_special_tokens': ['<bot>:', '<user>:', '|']})




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#quantization = GPTQConfig(bits=4, dataset='c4', tokenizer=tokenizer)
model = AutoModelForCausalLM.from_pretrained("gpt2_student")
#model = AutoModelForCausalLM.from_pretrained("datificate/gpt2-small-spanish", quantization_config=quantization)
#model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
#model = AutoModelWithLMHead.from_pretrained("ncoop57/DiGPTame-medium")


'''
# Obtener todos los parámetros (pesos) del modelo
params = list(model.parameters())


# Obtener una muestra de pesos
num_samples = 766  # Puedes ajustar este valor según tus necesidades

for p in params:
    print(p.numel())
    rand_index = random.sample(range(p.numel()), num_samples)
    weight_samples = p.data.cpu().numpy().flatten()[rand_index]

    # Crear un histograma de los valores de los pesos
    plt.hist(weight_samples, bins=50, color='blue', alpha=0.7)
    plt.title('Histograma de Pesos de GPT-2')
    plt.xlabel('Valor del Peso')
    plt.ylabel('Frecuencia')
    plt.grid(True)

    plt.show()
'''

tokenized_train_dataset, tokenized_test_dataset, tokenizer, model = TDUC.pipeline(root_dir, 2, tokenizer, model)




#peft_config = LoraConfig(task_type='CAUSAL_LM', r=1)
#model = get_peft_model(model, peft_config)
#model.print_trainable_parameters()

model.to(device)

args = {
    'nsamples': 128,
    'sparsity_ratio': 0.5,
    'seed': 70,
    'use_variant': False
}
#FT.prune_magnitude(args, model, tokenizer, device)
#FT.check_sparsity(model)
#FT.compressModel(model)
#LR 1Parte = 1e-5 (Probar a aumentar)
first_training_args = TrainingArguments(output_dir="results/FT_1", evaluation_strategy="epoch",
                                        per_device_train_batch_size=2, gradient_accumulation_steps=4,
                                        num_train_epochs=40,
                                        overwrite_output_dir=True, save_total_limit=1,
                                        load_best_model_at_end=True, save_strategy='epoch',
                                        lr_scheduler_type='cosine_with_restarts',
                                        learning_rate=1e-4, fp16=True,
                                        )

second_training_args = TrainingArguments(output_dir="results/FT_2", evaluation_strategy="epoch",
                                         per_device_train_batch_size=2, gradient_accumulation_steps=4,
                                         num_train_epochs=40,
                                         overwrite_output_dir=True, save_total_limit=1,
                                         load_best_model_at_end=True, save_strategy='epoch',
                                         lr_scheduler_type='cosine_with_restarts',
                                         learning_rate=5e-5, fp16=True,
                                         )


'''
num_blocks = 10
lr_list = [1e-1]
for i in range(10):
    unique_train_args = TrainingArguments(output_dir=f'IncrementalLR{i+1}', evaluation_strategy="epoch",
                                          per_device_train_batch_size=2,
                                          gradient_accumulation_steps=4, num_train_epochs=20,
                                          overwrite_output_dir=True, save_total_limit=1,
                                          load_best_model_at_end=True, save_strategy='epoch',
                                          lr_scheduler_type='cosine_with_restarts',
                                          fp16=True,
                                          )

    lr_list.append(lr_list[i]/10)
    FT.differentialLR(model, unique_train_args,'AdamW', lr_list,tokenizer, tokenized_train_dataset, tokenized_test_dataset)
    tokenizer.save_pretrained(f'IncrementalLR{i+1}/tokenizer')
'''

unique_train_args = TrainingArguments(output_dir=f'student_train', evaluation_strategy="epoch",
                                          per_device_train_batch_size=2, num_train_epochs=20,
                                          overwrite_output_dir=True, save_total_limit=1,
                                          load_best_model_at_end=True, save_strategy='epoch',
                                          lr_scheduler_type='cosine_with_restarts',
                                          fp16=True, learning_rate=1e-5
                                          )

trainer = Trainer(
    model=model,
    args=unique_train_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_test_dataset,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False
trainer.train()


'''
encoding = tokenizer('Hola', return_tensors='pt')
with torch.inference_mode():
    outputs = model.generate(
        input_ids = encoding.input_ids,
        attention_mask = encoding.attention_mask
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))


# Let's chat for 5 lines
for step in range(5):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(input(">> User:") + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens,
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # pretty print last ouput tokens from bot
    print("DialoGPT: {}".format(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)))
'''