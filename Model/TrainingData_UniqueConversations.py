import GenerateDataset as GD
import random
from datasets import Dataset
tokenizer = None

def pipeline(root_dir, blocks_per_conv, tokenizer, model):

    tokenizer = tokenizer
    def tokenizeFunction(examples):
        return tokenizer(examples["Conversations"], padding="max_length", truncation=True, max_length=200)

    Data = GD.getRawDataset2(root_dir, blocks_per_conv)
    num_train = int(0.9 * len(Data))
    new_vocab = GD.createVocabulary(
        [s.replace('<user>:', '').replace('<bot>:', '').replace('|', '') for s in Data])

    Data_train = random.sample(Data, num_train)
    Data_test = [test_sample for test_sample in Data if test_sample not in Data_train]

    train_dict = {"Conversations": Data_train}
    test_dict = {"Conversations": Data_test}

    train_dataset = Dataset.from_dict(train_dict)
    test_dataset = Dataset.from_dict(test_dict)

    new_vocab = set(new_vocab) - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(new_vocab))

    model.resize_token_embeddings(len(tokenizer))

    train_tokenized_dataset = train_dataset.map(tokenizeFunction, batched=True)
    test_tokenized_dataset = test_dataset.map(tokenizeFunction, batched=True)


    return train_tokenized_dataset, test_tokenized_dataset, tokenizer, model