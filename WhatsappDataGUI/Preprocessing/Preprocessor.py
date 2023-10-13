
def removeExtraMetadata(line: str):
    '''
    Removes Date and hour of a message. Only remains the message's author and the message itself.
    :param line: Whatsapp complete message
    :return: str of the form <Author>: <Message>
    '''
    return ' '.join(line.split()[3:])
