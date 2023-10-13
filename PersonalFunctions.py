import emoji

import Cleaning as C
import Analysis as A
import DataStructure as DS
import Plotting as P
from collections import Counter


def createGeneralWordcloud(df_chat, stopwords, store_path):
    """
    Creates Wordcloud from the chat
    :param df_chat: Pandas DF chat
    :param stopwords: list of stopwords
    :param store_path: str indicating where to allocate wordcloud image
    """

    df_chat = C.removeGroupNotification(df_chat)
    df_chat = C.removeMultimedia(df_chat)
    df_chat = C.removeErasedMessages(df_chat)
    df_chat = C.removeLostVideoCalls(df_chat)
    df_chat = C.removeLostGroupalVideoCalls(df_chat)
    df_chat = C.lowCase(df_chat)
    df_chat = C.removeStopwords(df_chat, stopwords)
    df_chat = C.emoji2text(df_chat)
    df_chat = C.removeEmojis(df_chat)
    df_chat = C.removeURL(df_chat)
    df_chat = C.removeAccents(df_chat)
    df_chat = C.removeLaughs(df_chat)

    chat = DS.dfToRaw(df_chat['message'], lowerize=True)

    A.createWordcloud(chat, store_path, isPersonal=False)


def createPersonalWordcloud(df_chat, username, stopwords, store_path):
    """
    Creates Wordcloud from the chat based on user messages.
    :param df_chat: Pandas DF chat
    :param username: str user to generate wordcloud from
    :param stopwords: list of stopwords
    :param store_path: str indicating where to allocate wordcloud image
    """

    new_chat = df_chat[df_chat['user'] == username]

    if len(new_chat) != 0:
        new_chat = C.removeGroupNotification(new_chat)
        new_chat = C.removeMultimedia(new_chat)
        new_chat = C.removeErasedMessages(new_chat)
        new_chat = C.removeLostVideoCalls(new_chat)
        new_chat = C.removeLostGroupalVideoCalls(new_chat)
        new_chat = C.lowCase(new_chat)
        new_chat = C.removeStopwords(new_chat, stopwords)
        new_chat = C.emoji2text(new_chat)
        new_chat = C.removeEmojis(new_chat)
        new_chat = C.removeURL(new_chat)
        new_chat = C.removeAccents(new_chat)
        new_chat = C.removeLaughs(new_chat)


        chat = DS.dfToRaw(new_chat['message'], lowerize=True)

        A.createWordcloud(chat, store_path, isPersonal=True)


def topKMostUsedEmojis(df_chat, K, wantPlot):
    """
    Returns the K most used emojis per user, if the user has used less emojis than
    K, then it returns all emojis used
    :param df_chat: Pandas DF that is the chat
    :param K: int Number of emojis to display per user
    :param wantPlot: boolean that indicates if plot needs to be generated.
    :return: Dictionary of K most used emojis per user
    """

    df_chat = C.emoji2text(df_chat)
    emoji_dict = emoji.UNICODE_EMOJI['en']
    symbols = list(emoji_dict.keys())
    words = list(emoji_dict.values())
    emojis_per_user = A.usedEmojis(df_chat)
    emojis_per_user = {k: v for k, v in emojis_per_user.items() if v} #Removing users with no emojis
    for user in emojis_per_user.keys():
        c = Counter({symbols[words.index(k)]: v for k, v in sorted(emojis_per_user[user].items(), key=lambda item: item[1])})
        emojis_per_user[user] = dict(c.most_common(K))

    if wantPlot:
        P.barPlotNestedDict(emojis_per_user, f'Top {K} emojis más usados por usuario.')

    return emojis_per_user


def multimediaSentPerUser(df_chat):
    """
    Returns number of multimedia sent per user
    :param df_chat: Pandas DF that is the chat
    :return: dictionary in the form: 'username': num_messages
    """

    return A.numMultimediaPerUSer(df_chat)

def averageWordsPerMessage(df_chat):
    """
    Returns, in average, how many words each message contains per user
    :param chat_df: Pandas DF chat
    :return: dictionary in the form: 'username': average lentgh of messages sent
    """

    df_chat = C.removeGroupNotification(df_chat)
    df_chat = C.emoji2text(df_chat)
    df_chat = C.removeErasedMessages(df_chat)
    df_chat = C.removeLostVideoCalls(df_chat)
    df_chat = C.removeLostGroupalVideoCalls(df_chat)
    df_chat = C.lowCase(df_chat)


    return A.averageNumWordsPerMessage(df_chat)

def messageContributionToChat(df_chat, wantPlot):
    """
    Computes in percentage the contribution in number of messages of each user in the chat.
    It can plot a pie plot if wanted.
    :param chat_df: Pandas DF chat
    :param wantPlot: boolean that indicates if plot needs to be generated.
    :return: dictionary in the form: 'username': rate_contribution
    """
    df_chat = C.removeGroupNotification(df_chat)
    df_chat = C.removeErasedMessages(df_chat)
    df_chat = C.removeLostVideoCalls(df_chat)
    df_chat = C.removeLostGroupalVideoCalls(df_chat)
    df_chat = C.emoji2text(df_chat)
    result = A.numMessagesPercentPerUSer(df_chat)
    if wantPlot:
        P.pieChart(result, 'Contribución de número de mensajes en tanto por ciento de cada usuario.')

    return result



def messagesAmountPerDay(df_chat):
    """
    Function computes number of messages per day. In total amount and per user.
    :param chat_df: Pandas DF, the chat.
    :return: A Tuple(Dict(Day: total_num_messages), Dict(UserX: Dict(Day: num_messages)))
    """
    df_chat = C.removeGroupNotification(df_chat)
    df_chat = C.removeErasedMessages(df_chat)
    df_chat = C.removeLostVideoCalls(df_chat)
    df_chat = C.removeLostGroupalVideoCalls(df_chat)

    return A.countMessagesPerDay(df_chat)


def kBiTriGramCount(df_chat, stopwords, k):
    """
    This function computes the top k most used bigrams and trigrams merged.
    :param df_chat: Pandas DF, the chat
    :param stopwords: list of stopwords
    :param k: Number of K most used Bigram/Trigrams
    :return: Pandas DF containing top most used Bigram/Trigram
    """

    df_chat = C.removeGroupNotification(df_chat)
    df_chat = C.removeMultimedia(df_chat)
    df_chat = C.removeErasedMessages(df_chat)
    df_chat = C.removeLostVideoCalls(df_chat)
    df_chat = C.removeLostGroupalVideoCalls(df_chat)
    df_chat = C.emoji2text(df_chat)
    df_chat = C.removeEmojis(df_chat)
    df_chat = C.lowCase(df_chat)
    df_chat = C.removeStopwords(df_chat, stopwords)
    df_chat = C.removeURL(df_chat)
    df_chat = C.removeLaughs(df_chat)


    bitri = A.usedBiTriGrams(df_chat)

    return bitri.iloc[0:k]