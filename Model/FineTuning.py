from transformers import Trainer, AdamW
import transformers
from boltons import iterutils
import random
import torch
import torch.nn as nn
from datasets import load_dataset


class GenerationCallback(transformers.TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 500 == 0:

          input_text = self.test_dataset['Conversations'][random.randint(0, len(self.test_dataset)-1)].split('|')[0]+' | <bot>:'
          print(f'Input Text: {input_text}')
          input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
          generated_ids = self.trainer.model.generate(input_ids=input_ids.to('cuda'),
                                         do_sample=True,
                                         max_new_tokens=20,
                                         num_beams=10,
                                         top_p=0.97,
                                         top_k=20,
                                         no_repeat_ngram_size=2,
                                         pad_token_id=self.tokenizer.pad_token_id)
          generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
          print("Generated Text:", generated_text)


def canFreezeLayers(model, condition, freeze):
    ct = 0
    for child in model.base_model.children():
        ct += 1
        print(ct)
        # print(child)
        if condition(ct):
            for name, param in child.named_parameters():
                print(f'{name} --> {param.numel()}')
                param.requires_grad = not freeze
    return model

def freezeExceptPredictorHead(model, first_train_arguments, second_train_arguments, tokenizer, train_dataset, test_dataset):

    model = canFreezeLayers(model, lambda id: id>1, True)

    trainer = Trainer(
        model=model,
        args=first_train_arguments,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    model.config.use_cache = False
    trainer.train()

    model = canFreezeLayers(model, lambda id: id>1, False)

    trainer = Trainer(
        model=model,
        args=second_train_arguments,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    model.config.use_cache = False
    trainer.train()


def groupingParams(param_list, lr_list):
    final_list = []
    window_size = int(len(param_list) / len(lr_list))
    ct = 0
    for i in range(len(lr_list)):
        sublist = param_list[ct:ct+window_size]
        final_list.append(sublist)
        ct += window_size

    return final_list


def differentialLR(model, train_args, opt_str, lr_list, tokenizer, train_dataset, test_dataset):
    params = [v for (k, v) in model.named_parameters()]

    if opt_str == 'AdamW':
        optimizer = AdamW([{'params':params[idx], 'lr': lr_list[idx]} for idx in range(len(groupingParams(params, lr_list)))],
                          lr=1e-3)

    callback = GenerationCallback()

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        optimizers=(optimizer, None)
    )

    callback.trainer = trainer
    callback.tokenizer = tokenizer
    callback.test_dataset = test_dataset
    trainer.add_callback(callback)
    model.config.use_cache = False
    trainer.train()

def gradualUnfreeze(model, train_args, lr_list, tokenizer, train_dataset, test_dataset):
    model = canFreezeLayers(model, lambda id: id < 1, True)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    model.config.use_cache = False
    trainer.train()

    return







def prepare_calibration_input(model, dataloader, device):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h


    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((len(dataloader), 20, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            #cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids

def get_layers(block):
    layers = {}
    for name, module in block.named_children():
       if 'ln' in name:
           layers[name] = module
       else:
           for name2, module2 in module.named_children():
               if hasattr(module2, 'weight'):
                    layers[name2] = module2

    return layers

def get_layers2(model):
    layers = {}
    for child in model.base_model.children():
        for name, param in child.named_parameters():
                layers[name] = param
                print(f'{name} --> {param.numel()}')
    return layers

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

# Define WrappedGPT class
class WrappedGPT:
    """
    This class wraps a GPT layer for specific operations.
    """

    def __init__(self, layer, layer_id=0, layer_name="none"):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.rows = layer.weight.data.shape[0]


        if len(layer.weight.data.shape) > 1:
            self.columns = layer.weight.data.shape[1]
            self.scaler_row = torch.zeros((self.columns), device=self.dev)
        else:
            self.columns = 1
            self.scaler_row = torch.zeros((self.rows), device=self.dev)
            self.scaler_row = self.scaler_row.unsqueeze(0)
        self.nsamples = 0

        self.layer_id = layer_id
        self.layer_name = layer_name



    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()

        self.scaler_row *= self.nsamples / (self.nsamples+tmp)
        self.nsamples += tmp

        inp = inp.type(torch.float32)

        self.scaler_row += torch.norm(inp, p=2, dim=1) ** 2  / self.nsamples


# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

def get_loader(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    traindata = load_dataset('amazon_reviews_multi', 'es', data_files={'train': 'json/train/dataset_es_train.json'}, split='train')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, len(traindata) - 1)
        trainenc = tokenizer(traindata[i]['review_body'], return_tensors='pt', truncation=True, max_length=seqlen, padding='max_length')
        trainloader.append(trainenc['input_ids'])

    return trainloader

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader = get_loader(nsamples=args['nsamples'],seed=args['seed'],seqlen=20,tokenizer=tokenizer)
    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device)

    layers = model.transformer.h
    for i in range(len(layers)):
        layer = layers[i]
        subset = get_layers(layer)


        wrapped_layers = {}
        for name in subset:
            if 'dropout' not in name and 'act' not in name:
                wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args['nsamples']):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0))[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args['use_variant']:
                    # wanda variant
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args['sparsity_ratio'])>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args['sparsity_ratio']:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args['sparsity_ratio'])]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero

        for j in range(args['nsamples']):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()

def custom_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader = get_loader(nsamples=args['nsamples'], seed=args['seed'], seqlen=20, tokenizer=tokenizer)
    print("dataset loading complete")

    for element in dataloader:
        print(element)

def check_sparsity(model):
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.transformer.h
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = get_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache
    return float(count)/total_params

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.transformer.h

    for i in range(len(layers)):
        block = layers[i]
        subset = get_layers(block)

        for name in subset:
            W = subset[name].weight.data
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args['sparsity_ratio'])].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def compressModel(model):
    layers = model.transformer.h

    for i in range(len(layers)):
        block = layers[i]

        for name, module in block.named_children():
            if 'ln' in name:
                module.weight = torch.nn.Parameter(module.weight.data.to_sparse())
            else:
                for name2, module2 in module.named_children():
                    if hasattr(module2, 'weight'):
                        module2.weight = torch.nn.Parameter(module2.weight.data.to_sparse())

    return layers
