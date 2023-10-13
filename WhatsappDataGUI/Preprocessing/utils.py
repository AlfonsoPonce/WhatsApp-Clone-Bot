import re

def addCustomLabel(preprocessed_line):
    '''
    Adds a label to a line as <user>Message<user>. Calling removeExtraMetadata MUST be done.
    :param preprocessed_line: Message with the form <User>: <Message>
    :return: Modified line with the user label. If user is a number, a generic label is set.
    '''
    telephone_regexp = r"\+34"

    if re.search(telephone_regexp, preprocessed_line):
        if '<context>' in preprocessed_line:
            index_to_add = preprocessed_line.index('|')
            return f'{preprocessed_line[:index_to_add+1]}<GenericUser> {preprocessed_line[index_to_add+1:]} <GenericUser>'
        else:
            return f'<GenericUser> {preprocessed_line} <GenericUser>'
    else:
        if '<context>' in preprocessed_line:
            author = preprocessed_line.split('|')[1].split(':')[0]
            index_to_add = preprocessed_line.index('|')
            return f'{preprocessed_line[:index_to_add+1]}<{author}> {preprocessed_line[index_to_add+1:]} <{author}>'
        else:
            author = preprocessed_line.split(':')[0]  # We dont want the colon in user's name
            return f'<{author}> {preprocessed_line} <{author}>'


def eraseCustomLabel(preprocessed_line):
    if '<context>' in preprocessed_line:
        context = preprocessed_line.split('|')[0]
        if '<Multimedia omitido>' in preprocessed_line: #Change expression for another language
            expression_list = re.findall(r'<[^>]*>|[^<]+', ' '.join(preprocessed_line.split('|')[1:]))
            message = expression_list[1][1:] + expression_list[2]
            return f'{context}|{message}'
        else:
            message = ' '.join(re.findall(r'<[^>]*>|[^<]+', ' '.join(preprocessed_line.split('|')[1:]))[1:-1])[1:-1]
            return f'{context}|{message}'
    else:
        if '<Multimedia omitido>' in preprocessed_line: #Change expression for another language
            expression_list = re.findall(r'<[^>]*>|[^<]+', preprocessed_line)
            return expression_list[1][1:] + expression_list[2]
        else:
            return ' '.join(re.findall(r'<[^>]*>|[^<]+', preprocessed_line)[1:-1])[1:-1]

def addContext(context, current_message):
    return f'<context> {context} <context>|{current_message}'

def eraseContext(current_message):
    if len(current_message.split('|')) == 2:
        return ' '.join(current_message.split('|')[1:])
    else:
        return current_message