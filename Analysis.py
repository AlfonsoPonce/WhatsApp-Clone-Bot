'''
This Python file is intended for perform text analytics such as word cloud, frequent words, etc.
'''
import datetime
import re
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import emoji
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def numEmojis(chat_df):
    """
    Counts how many emojis a chat has.
    :param chat: str of demojized chat
    :return: number of emojis in that chat
    """
    n_emojis = 0
    emoji_dict = emoji.UNICODE_EMOJI['en']
    for message in chat_df['message'].values:
        n_emojis += len([w for w in message.split() if w in emoji_dict.values()])

    return n_emojis



def createWordcloud(chat, store_path, isPersonal):
    """
    Creates Wordcloud from the chat
    :param df_chat: Pandas DF chat
    :param stopwords: list of stopwords
    :param isPersonal: True if it is a personal wordcloud, False if general Wordcloud
    :param store_path: str indicating where to allocate wordcloud image
    """
    # Create and generate a word cloud image:
    wordcloud = WordCloud(background_color='white').generate(chat)

    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(store_path.name)
    plt.axis("off")
    if isPersonal:
        plt.savefig(store_path.joinpath('PersonalWordcloud.png'))
    else:
        plt.savefig(store_path.joinpath('GeneralWordcloud.png'))


def numMessagesPerUSer(chat_df):
    """
    Returns how many messages each participant has sent into the chat.
    :param chat_df: Pandas DF chat
    :return: dictionary in the form: 'username': num_messages
    """

    new_df = chat_df.groupby(['user']).size().reset_index(name='counts')
    results = {}
    for user_stats in new_df.values:
        if user_stats[0] != 'group_notification':
            results[user_stats[0]] = user_stats[1]

    return results

def numMessagesPercentPerUSer(chat_df):
    """
    Returns relative frequency of messages from each participants in percent rate
    :param chat_df: Pandas DF chat
    :return: dictionary in the form: 'username': rate_contribution
    """

    results = numMessagesPerUSer(chat_df)

    total_messages = sum(list(results.values()))
    for user, value in results.items():
        results[user] = round(value/total_messages*100)

    return results

def averageNumWordsPerMessage(chat_df):
    """
    Returns, in average, how many words each message contains per user
    :param chat_df: Pandas DF chat
    :return: dictionary in the form: 'username': average lentgh of messages sent
    """

    total_message_per_user = numMessagesPerUSer(chat_df)
    dict_average = {}

    for _, user, _ in chat_df.values:
        average = 0
        if user != 'group_notification':
            new_df = chat_df[chat_df['user'] == user]
            for _,_,message in new_df.values:
                average += len(message.split())

            average /= total_message_per_user[user]
            dict_average[user] = round(average)

    return dict_average



def numMultimediaPerUSer(chat_df):
    """
    Returns how many multimedia messages each participant has sent into the chat.
    :param chat_df: Pandas DF chat
    :return: dictionary in the form: 'username': num_messages
    """

    new_df = chat_df[chat_df['message'] == '<Multimedia omitido> '].groupby(['user']).size().reset_index(name='counts')
    results = {}
    for user_stats in new_df.values:
        if user_stats[0] != 'group_notification':
            results[user_stats[0]] = user_stats[1]

    return results
#TODO Corregir y acabar funnction
def maxTimeToAnswer(chat_df):
    """
    Computes max time a person lasts to answer
    :param chat_df: Pandas DF that is the chat
    :return: Dict with the form 'user': max_time_to_answer
    """
    new_chat = chat_df.drop(chat_df[chat_df['user']== 'group_notification'].index)
    new_chat = new_chat.groupby(['date_time', 'user']).count()
    isGroup = 'grupo' in chat_df['message'].iloc[0]
    result = {}
    if not isGroup:
        prev_user = new_chat.index[0][1]
        prev_date = new_chat.index[0][0]
        max_dif = datetime.timedelta()
        max_user = ''
        for curr_date, curr_user in new_chat.index:
            if prev_user != curr_user:
                dif = curr_date - prev_date
                if dif > max_dif:
                    max_dif = dif
                    max_user = curr_user
                prev_user = curr_user
                prev_date = curr_date

        return max_user, max_dif


def usedEmojis(chat_df):
    """
    Returns number of emojis used in chat
    :param chat_df: Pandas DF that is the chat
    :return: Dict with the form 'user': {'emoji': times_used,...}
    """
    emoji_dict = {}
    emojis = emoji.UNICODE_EMOJI['en']
    for _,user,message in chat_df.values:
        if user != 'group_notification':
            emoji_per_message_list = [w for w in message.replace('::', ': :').split() if w in emojis.values()]
            ocurrences = dict(Counter(emoji_per_message_list))
            if user not in emoji_dict.keys():
                emoji_dict[user] = ocurrences
            else:
                for e in ocurrences.keys():
                    if e not in emoji_dict[user].keys():
                        emoji_dict[user][e] = ocurrences[e]
                    else:
                        emoji_dict[user][e] += ocurrences[e]


    return emoji_dict


def countMessagesPerDay(chat_df):
    """
    Function computes number of messages per day. In total amount and per user.
    :param chat_df: Pandas DF, the chat.
    :return: A Tuple(Dict(Day: total_num_messages), Dict(UserX: Dict(Day: num_messages)))
    """
    global_result = {}
    per_user_result = {}
    for user in chat_df['user'].unique().tolist():
        per_user_result[user] = {}
    new_df = chat_df.copy()
    new_df['date_time'] = chat_df['date_time'].apply(lambda x: str(x).split()[0])
    count_df = new_df.groupby(['date_time'])['date_time'].count()
    indexes = count_df.index
    for i in indexes:
        global_result[i] = count_df.loc[i]

    for date in indexes:
        for user in per_user_result.keys():
            per_user_result[user][date] = 0

    user_count_df = new_df.groupby(['date_time', 'user'])['date_time'].count()
    user_indexes = user_count_df.index

    for i in user_indexes:
        date = i[0]
        user = i[1]
        per_user_result[user][date] = user_count_df.loc[i]


    return (global_result, per_user_result)

def usedBiTriGrams(df_chat):
    """
    This function computes the used bigrams and trigrams merged.
    :param df_chat: Pandas DF, the chat
    :return: Pandas DF containing used Bigram/Trigram
    """

    c_vec = CountVectorizer(ngram_range=(2, 3))
    # matrix of ngrams
    ngrams = c_vec.fit_transform(df_chat['message'])
    # count frequency of ngrams
    count_values = ngrams.toarray().sum(axis=0)
    # list of ngrams
    vocab = c_vec.vocabulary_

    df_ngram = pd.DataFrame(sorted([(count_values[i], j) for j, i in vocab.items()], reverse=True)
                            ).rename(columns={0: 'frequency', 1: 'bigram/trigram'})

    return df_ngram