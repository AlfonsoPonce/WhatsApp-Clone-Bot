import GenerateDataset as GD
from sklearn.model_selection import train_test_split
from datasets import Dataset

tokenizer = None



def pipeline(root_dir, tokenizer, model):
    tokenizer = tokenizer

    def tokenizeInput(examples):
        return tokenizer(examples["Input"], padding="max_length", truncation=True, max_length=50)

    def tokenizeOutput(examples):
        return tokenizer(examples["Output"], padding="max_length", truncation=True, max_length=50)

    X, Y = GD.getRawDataset(root_dir)
    X, Y = GD.eraseEmptyQueries(X, Y)

    new_vocab = GD.createVocabulary(
        [s.replace('<user>:', '').replace('<bot>:', '').replace('<turn>', '') for s in X + Y])

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.1)

    train_dict = {'Input': X_train, 'Output': y_train}
    test_dict = {'Input': X_test, 'Output': y_test}
    train_dataset = Dataset.from_dict(train_dict)
    test_dataset = Dataset.from_dict(test_dict)
    new_vocab = set(new_vocab) - set(tokenizer.vocab.keys())
    tokenizer.add_tokens(list(new_vocab))

    model.resize_token_embeddings(len(tokenizer))

    train_tokenized_dataset = train_dataset.map(tokenizeInput, batched=True)
    train_tokenized_dataset = train_tokenized_dataset.map(tokenizeOutput, batched=True)

    test_tokenized_dataset = test_dataset.map(tokenizeInput, batched=True)
    test_tokenized_dataset = test_tokenized_dataset.map(tokenizeOutput, batched=True)

    return train_tokenized_dataset, test_tokenized_dataset, tokenizer, model

