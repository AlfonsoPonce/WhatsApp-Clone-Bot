import re
import pandas as pd



def rawToDf(raw_data, key):
    '''Converts raw .txt file into a Data Frame'''

    split_formats = {
        '12hr': '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s[APap][mM]\s-\s',
        '24hr': '\d{1,2}/\d{1,2}/\d{2,4},\s\d{1,2}:\d{2}\s-\s',
        'custom': ''
    }
    datetime_formats = {
        '12hr': '%d/%m/%Y, %I:%M %p - ',
        '24hr': '%d/%m/%y, %H:%M - ',
        'custom': '%d/%m/%y, %H:%M - '
    }

    raw_string = ' '.join(raw_data.split(
            '\n'))  # converting the list split by newline char. as one whole string as there can be multi-line messages
    user_msg = re.split(split_formats[key], raw_string)[
                   1:]  # splits at all the date-time pattern, resulting in list of all the messages with user names
    date_time = re.findall(split_formats[key], raw_string)  # finds all the date-time patterns

    df = pd.DataFrame({'date_time': date_time, 'user_msg': user_msg})  # exporting it to a df

    # converting date-time pattern which is of type String to type datetime,
    # format is to be specified for the whole string where the placeholders are extracted by the method
    df['date_time'] = pd.to_datetime(df['date_time'], format=datetime_formats[key])

    # split user and msg
    usernames = []
    msgs = []
    for i in df['user_msg']:
        a = re.split('([\w\W]+?):\s',
                     i)  # lazy pattern match to first {user_name}: pattern and spliting it aka each msg from a user
        if (a[1:]):  # user typed messages
            usernames.append(a[1])
            msgs.append(a[2])
        else:  # other notifications in the group(eg: someone was added, some left ...)
            usernames.append("group_notification")
            msgs.append(a[0])

    # creating new columns
    df['user'] = usernames
    df['message'] = msgs

    # dropping the old user_msg col.
    df.drop('user_msg', axis=1, inplace=True)

    return df

def dfToRaw(serie, lowerize):
    """
    Converts dataframe to string
    :param serie: column of dataframe that are messages
    :param lowerize: if True, lowercase all letters
    :return: str chat
    """
    comment_words = ' '
    for val in serie.values:

        # typecaste each val to string.
        val = str(val)

        # split the value.
        tokens = val.split()

        if lowerize:
            # Converts each token into lowercase.
            for i in range(len(tokens)):
                tokens[i] = tokens[i].lower()

        for words in tokens:
            comment_words = comment_words + words + ' '

    return comment_words


def __blockMessage__(df, curr_user, curr_index):
    block = ''
    while df['user'].iloc[curr_index] == curr_user:
        block = block + df['message'].iloc[curr_index] + ' \n '
        curr_index = curr_index + 1
        if curr_index >= len(df):
            break
    curr_index = curr_index - 1
    return block, curr_index

def __eraseFirstMessages(df, user_to_replicate):
    num_m = len(df)
    new_df = df.copy().reset_index().drop('index', axis=1)
    i = 0
    while i < num_m:
        if new_df['user'].iloc[i] == user_to_replicate:
            new_df = new_df.drop(i)
        else:
            break
        i += 1
    return new_df
def dfToTrain(chat_df: pd.DataFrame, user_to_replicate: str):
    """
    This function tries to automatize the creation of a training dataset, it is not accurate
    :param chat_df:
    :param user_to_replicate: string that will indicate messages to be target
    :return: Pandas DF with query-response format
    """
    new_df = chat_df.drop('date_time', axis=1)
    new_df = new_df.drop(new_df[new_df['user'] == 'group_notification'].index)
    if new_df['user'].iloc[0] == user_to_replicate:
        new_df = __eraseFirstMessages(new_df, user_to_replicate)
    queries = []
    responses = []
    i = 0
    replicated_user_answered = False
    while i < len(new_df):
        if i+1 < len(new_df):
            messages, i = __blockMessage__(new_df, new_df['user'].iloc[i], i)
            if new_df['user'].iloc[i] != user_to_replicate:
                queries.append(messages)
            else:
                responses.append(messages)
                replicated_user_answered=True
            if i+1 < len(new_df):
                if new_df['user'].iloc[i+1] != user_to_replicate:
                    if not replicated_user_answered:
                        responses.append('#')             #Not entering here in case of individual chat
                    replicated_user_answered = False
            else:
                responses.append('#')



        i += 1

    return pd.DataFrame(list(zip(queries, responses)), columns=['Query', 'Response'])


def __individualChatModeling(df: pd.DataFrame):
    # Calculate time passed since previous message
    df["date_previous"] = df["date"].shift(periods=1)
    df["time_delta"] = (df["date"] - df["date_previous"]).dt.total_seconds()

    # Concat message and author
    df["chat_message"] = df["username"] + ": " + df["message"]

    # Remove first line, its just a WhatApp test line
    df = df[1:]

    # Convert messages into conversations (a conversation has multiple messages); ugly programming, but works for small data
    # Step 1: Concat each message with the previous conversation
    query = []
    answer = []
    conversation = ""
    session_ix = 0
    sessions_ixs = []

    for ix, row in df.iterrows():
        if row["time_delta"] < 3600:  # This defines on how close messages should be to be in the same conversation
            session_ix = session_ix + 1
            sessions_ixs.append(session_ix)
            if conversation == "":
                conversation = row["chat_message"]
                query.append(conversation)
                answer.append("")
            else:
                conversation = conversation + "| " + row["chat_message"]
                query.append(conversation)
                answer.append(row["chat_message"])
        else:
            session_ix = 0
            conversation = ""

    df_model = pd.DataFrame({"query": query[:-1], "answer": answer[1:], "session_ix": sessions_ixs[:-1]})

    # Step 2: Filter only for the last message of the conversation (therefore for the full conversation.
    df_model["model_helper_idx"] = df_model["session_ix"] - df_model["session_ix"].shift(-1)
    df_model = df_model[df_model["model_helper_idx"] > -1]

    # This way is a bit clumsy, but I did some test with the intermediate conversation steps.

    return df_model


def dfToTrainGPT2(chat_df: pd.DataFrame, user_to_replicate: str):
    is_group = len(chat_df['username'].unique()) > 2
    if not is_group:
        result = __individualChatModeling(chat_df)
    return
