'''
This Python file is intended for perform text cleaning such that emoji removal, stopword removal, stemming, etc.
'''
import re

import pandas as pd
import spacy
import emoji

def emoji2text(chat_df):
    """
    Function to transform emojis into text.
    :param chat: Dataframe object that is the chat.
    :return: str chat whose emojis are turned into text
    """
    for i in range(len(chat_df['message'].values)):
        chat_df['message'].iloc[i] = emoji.demojize(chat_df['message'].iloc[i])
    return chat_df

def removeEmojis(chat_df):
    """
    Function erase emojis from chat. Calls emoji2text.
    :param chat: DF object that is the chat.
    :return: str chat without emojis
    """
    new_chat = emoji2text(chat_df)

    emoji_dict = emoji.UNICODE_EMOJI['en']
    for i in range(len(new_chat['message'].values)):
        new_chat['message'].iloc[i] = ' '.join([w for w in new_chat['message'].iloc[i].replace('::', ': :').split() if w not in emoji_dict.values()])
    return new_chat

#TODO expand to new signs, careful with ":" that indicates emojis
def removePunctuation(chat):
    """
    This function removes punctuation signs used in Spanish language.
    :param chat: str object that is the chat.
    :return: str chat with no punctuation signs (Spanish context)
    """

    new_chat = chat.replace(".","")
    new_chat = new_chat.replace(",", "")
    new_chat = new_chat.replace(";", "")
    new_chat = new_chat.replace("?", "")
    new_chat = new_chat.replace("¿", "")
    new_chat = new_chat.replace("!", "")
    new_chat = new_chat.replace("¡", "")

    return new_chat

def removeStopwords(chat_df, stopwords):
    """
    Function to erase stopwords from the chat
    :param chat:  str object that is the chat.
    :param stopwords:  list containing words to remove.
    :return: str chat with no punctuation signs
    """

    set_stop = set(stopwords)
    for i in range(len(chat_df['message'].values)):
        meaningful_words = [w for w in chat_df['message'].iloc[i].split() if not w in set_stop]
        chat_df['message'].iloc[i] = " ".join(meaningful_words)

    return chat_df

def lowCase(chat_df):
    """
    Turns all words in chat into lower case
    :param chat: str object that is the chat.
    :return: str chat with all words in low case
    """
    for i in range(len(chat_df['message'].values)):
        chat_df['message'].iloc[i] = chat_df['message'].iloc[i].lower()
    return chat_df

def tokenize(chat):
    """
    Tokenize chat
    :param chat: str object that is the chat.
    :return: list of words used in chat
    """

    return chat.split()

def removeMessageHeader(chat):
    """
    Removes Header from WhatsaApp Messages
    :param chat: str object that is the chat.
    :return: str chat without header
    """

    list_messages = chat.split('\n')
    for message in list_messages:
        l = message.split('-')
        n = " ".join(l[1:])
        n = n.split(':')
        final_message = " ".join(n[1:])
        list_messages[list_messages.index(message)] = final_message[1:]+'\n'

    return " ".join(list_messages)


def removeMessageHeaderExceptName(chat):
    """
    Removes Header from WhatsaApp Messages, keeping the user name
    :param chat: str object that is the chat.
    :return: str chat without header
    """
    list_messages = chat.split('\n')
    for message in list_messages:
        l = message.split('-')
        final_message = " ".join(l[1:])
        list_messages[list_messages.index(message)] = final_message[1:]

    return " ".join(list_messages)

#Cuidado con el MUltimedia, ya que puede haber mensajes posteriores que hagan alusión a ella sin responder
#nada anterior. O contestar a un mensaje con Multimedia.
def removeMultimedia(df: pd.DataFrame, is_from_whatstk: bool):
    """
    Removes Multimedia from chat (Spanish Context)
    :param chat: str object that is the chat.
    :return: str chat without Multimedia
    """
    if not is_from_whatstk:
        media = df[df['message'] == '<Multimedia omitido> ']
    else:
        media = df[df['message'] == '<Multimedia omitido>']
    df.drop(media.index, inplace=True)  # removing images
    df.reset_index(inplace=True, drop=True)

    return df

def removeURL(chat_df):
    """
    Removes Multimedia from chat (Spanish Context)
    :param chat: str object that is the chat.
    :return: str chat without Multimedia
    """
    regexp = r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))'''

    for i in range(len(chat_df['message'].values)):
        chat_df['message'].iloc[i] = re.sub(regexp, '', chat_df['message'].iloc[i])

    return chat_df

def lemmatization(chat_df):
    """
    Performs lemmatization on every message on chat
    :param chat: str that is the chat
    :return: lemmatized chat
    """

    nlp = spacy.load("es_core_news_lg")
    for i in range(len(chat_df['message'].values)):
        lemmatized = nlp(chat_df['message'].iloc[i])
        chat_df['message'].iloc[i] = " ".join([token.lemma_ for token in lemmatized])

    return chat_df



def removeExtraVocals(chat_df):
    """
    Removes extra vocals in words like "vaaalee" --> "vale"
    :param chat: str chat
    :return: str chat with extra vocals erased
    """
    for i in range(len(chat_df['message'].values)):
        new_chat = tokenize(chat_df['message'].iloc[i])

        for word in new_chat:
            new_word = word
            for letter in new_word:
                if new_word.index(letter) < len(new_word)-1:
                    if new_word[new_word.index(letter)] == new_word[new_word.index(letter)+1]:
                        new_word[new_word.index(letter)+1] = ''
            new_chat[new_chat.index(word)] = new_word
        chat_df['message'].iloc[i]= " ".join(new_chat)
    return chat_df


def removeLaughs(chat_df):
    """
    Removing laughs of the for "jajaja" or "JAJAJA"
    :param chat: str that is the chat
    :return: chat without laughs
    """

    regexp = r"\b(?:a*(?:ja)+j?|j*ja+j[ja]*|(?:J+A+)+J+|A?J+A+J+[AJ]*)\b"
    for i in range(len(chat_df['message'].values)):
        chat_df['message'].iloc[i] = re.sub(regexp, '', chat_df['message'].iloc[i])
    return chat_df


def removeGroupNotification(df_chat):
    """
    Removes group notifications
    :param df_chat: DF object that is the chat
    :return: Dataframe with no group notifications
    """
    new_chat = df_chat.drop(df_chat[df_chat['user'] == 'group_notification'].index)
    return new_chat

def removeAccents(df_chat):
    """
    Removes Accents
    :param df_chat: DF object that is the chat
    :return: Dataframe with accents
    """
    replacements = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
    )
    for i in range(len(df_chat['message'].values)):
        for a,b in replacements:
            df_chat['message'].iloc[i] = df_chat['message'].iloc[i].replace(a, b).replace(a.upper(), b.upper())

    return df_chat

def removeErasedMessages(df_chat):
    """
    This function removes messages that were erased
    :param df_chat: Pandas DF, the chat
    :return: Chat without erased messages
    """
    media = df_chat[df_chat['message'] == 'Se eliminó este mensaje.']
    df_chat.drop(media.index, inplace=True)  # removing images
    df_chat.reset_index(inplace=True, drop=True)

    return df_chat

def removeLostVideoCalls(df_chat):
    """
    This function removes lost videocalls
    :param df_chat: Pandas DF, the chat
    :return: Chat without lost videocalls
    """
    media = df_chat[df_chat['message'] == 'Videollamada perdida']
    df_chat.drop(media.index, inplace=True)  # removing images
    df_chat.reset_index(inplace=True, drop=True)

    return df_chat

def removeLostGroupalVideoCalls(df_chat):
    """
    This function removes lost groupal videocalls
    :param df_chat: Pandas DF, the chat
    :return: Chat without groupal videocalls
    """
    media = df_chat[df_chat['message'] == 'Videollamada grupal perdida']
    df_chat.drop(media.index, inplace=True)  # removing images
    df_chat.reset_index(inplace=True, drop=True)

    return df_chat