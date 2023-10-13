import pandas as pd




def getRawDataset2(root_dir, num_blocks_per_session):
    '''
    This function turns pairs of query-response (each query and each response can have more than one message --> a block)
    into a conversation. Can work with individual chats or groups.
    :param root_dir:
    :param num_blocks_per_session:
    :return:
    '''
    Sessions = list()
    for file in root_dir.glob('*csv'):
        print(file.name)
        df = pd.read_csv(str(file), encoding='utf-8')
        sess_idx_mssg = 0
        new_session = ''
        for i, data in df.iterrows():
            if data[1] != '#':
                if sess_idx_mssg < num_blocks_per_session or len(Sessions) == 0:
                    new_session = new_session + '<user>:' + data[0] + '| <bot>:' + data[1] + '|\n'
                    sess_idx_mssg += 1
                else:
                    Sessions.append(new_session+'<|endoftext|>')
                    new_session = ''
                    sess_idx_mssg = 0

        if new_session not in Sessions:
            Sessions.append(new_session+'<|endoftext|>')
    return Sessions


def flatten(l):
    return [item for sublist in l for item in sublist]

def getRawDataset(root_dir):
    '''
    Returns Data of Query-Response in the form of X-Y.
    :param root_dir:
    :return:
    '''
    X, Y = list(), list()
    for file in root_dir.glob('*csv'):
        print(file.name)
        df = pd.read_csv(str(file), encoding='utf-8')
        X.append([i for i in df['Query']])
        Y.append([i for i in df['Response']])

    X = flatten(X)
    Y = flatten(Y)

    return X,Y

def addSpecialTags(X,Y):
    for i,_ in enumerate(X):
        X[i] = '<user>:'+X[i]
    for i,_ in enumerate(Y):
        Y[i] = '<bot>:'+Y[i]

    return X,Y

def eraseEmptyQueries(X,Y):
    for answer in Y:
      if answer == '#':
        index = Y.index('#')
        Y.remove('#')
        del X[index]
    return X, Y

def createVocabulary(Y, X=None):
    if X == None:
      return set(flatten([s.split() for s in Y]))
    else:
      s1 = set(flatten([s.split() for s in Y]))
      s2 = set(flatten([s.split() for s in X]))
      return s1.union(s2)

def eraseMultimediaConversations(X, Y):
    for answer in Y:
      if '<Multimedia omitido>' in answer:
        index = Y.index(answer)
        Y.remove(answer)
        del X[index]
    return X, Y

