from pathlib import Path
import Cleaning as C
import Analysis as A
import DataStructure as DS
import PersonalFunctions as PF
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pprint import pprint
from dash import Dash, dcc, html,Input, Output
pd.options.mode.chained_assignment = None  # default='warn'
data_dir = Path('Capturas_23052023')
WORDCLOUD_DIR = Path('assets/wordclouds')
root_store_dir = Path('Analytics')
root_store_dir.mkdir(exist_ok=True)
#print('Introduce tu nombre de WhatsApp: ')

#username = input('>')
#print('Procesando toda la información del directorio Capturas/...')
#print('Esto puede tardar unos instantes...')
stopwords = open('WhatsApp_StopWords.txt', 'r', encoding='utf-8').read().split()
USERNAME = None
GLOBAL_DF = None
chats = [str(files.name).replace('.txt', '') for files in data_dir.glob('*.txt')]

app = Dash(__name__)



app.layout = html.Div(
    children=[
            html.Div(children=[html.H1(children="Estadísticas de los chats de"),html.Img(src='/assets/whatsapp_img.png')], className='title_image'),
            html.Div(children=[html.P("Selecciona el chat:"), dcc.Dropdown(id='names',options=chats,value=chats[1], clearable=False, className='chat_names_dd')], className='chat_selector'),
            html.Div(children=[html.H1(children="Contribución por usuario en mensajes"),dcc.Graph(id="graph")], className='contribution'),
            html.Div(children=[html.H1(children="Número de mensajes al día"),dcc.Graph(id="messages_day_graph")], className='contribution'),
            html.Div(children=[html.H1(children="Número de elementos multimedia enviados"),dcc.Graph(id="multimedia_graph")], className='contribution'),
            html.Div(children=[html.H1(children="Wordcloud del chat"),html.Img(id='global_wordcloud', src=str(WORDCLOUD_DIR.joinpath(chats[1]).joinpath('GeneralWordcloud.png')))], className='contribution'),
            html.Div(children=[html.H1(children="Expresiones de 2 y 3 palabras más usadas."),dcc.Graph(id="bitrigram_graph")], className='contribution'),
            html.Div(children=[
                html.H1(children="Top 5 Emojis más usados por los usuarios"),
                html.P(children="Selecciona el usuario"),
                dcc.Dropdown(id='other_names',clearable=False,value='hola'),
                dcc.Graph(id="emoji_graph"),

                        ], className='individual_info')

            ], className='app'
)

@app.callback(
    Output('other_names', 'options'),
    Input('names', 'value')
)
def getChatUsers(value):
    result = list(GLOBAL_DF[GLOBAL_DF['file'] == value]['data'].values[0]['user_contribution'].keys())

    return result

@app.callback(
    Output('multimedia_graph', 'figure'),
    Input("names", "value")
)
def generateMultimediaBarplot(value):
    fig = {}
    if len(GLOBAL_DF[GLOBAL_DF['file'] == value]['data'].values) != 0:
        result = GLOBAL_DF[GLOBAL_DF['file'] == value]['data'].values[0]['multimedia_sent']
        fig = px.bar(pd.DataFrame(result.items(), columns=['user', 'counts']), y='counts', x='user')

    return fig

@app.callback(
    Output('global_wordcloud', 'src'),
    Input('names', 'value')
)
def generateGlobalWordcloud(value):
    return str(WORDCLOUD_DIR.joinpath(value).joinpath('GeneralWordcloud.png'))


@app.callback(
    Output('bitrigram_graph', 'figure'),
    Input('names', 'value')
)
def generateGlobalBiTrigraph(value):
    result = GLOBAL_DF[GLOBAL_DF['file'] == value]['data'].values[0]['global_bitrigram']
    fig = px.bar(result, y='bigram/trigram', x='frequency', orientation='h')
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig

@app.callback(
    Output('messages_day_graph', 'figure'),
    Input("names", "value")
)
def generateMessagesPerDayTimeLine(value):
    global_data, per_user_data = GLOBAL_DF[GLOBAL_DF['file'] == value]['data'].values[0]['messages_per_day']
    fig = px.line(pd.DataFrame(global_data.items(), columns=['date', 'counts']), x='date', y='counts')
    fig.data[0].name = 'Total'
    fig.update_traces(showlegend=True)
    i = 1
    for user in per_user_data.keys():
        df = pd.DataFrame(per_user_data[user].items(), columns=['date', 'counts'])
        fig.add_scatter(x=df['date'], y=df['counts'])
        fig.data[i].name = user
        i+=1
    return fig

@app.callback(
    Output('emoji_graph', 'figure'),
    Output('other_names', 'value'),
    Input('names', 'value'),
    Input('other_names', 'value'),
    )
def generateHorizontalBarplot2(chat_value, names_value):
    fig = {}
    print(names_value)
    if names_value != None:
        if len(GLOBAL_DF[GLOBAL_DF['file']==chat_value]['data'].values) != 0:
            if GLOBAL_DF[GLOBAL_DF['file']==chat_value]['data'].values[0]['most_used_emojis'] != {}:
                result = GLOBAL_DF[GLOBAL_DF['file']==chat_value]['data'].values[0]['most_used_emojis'][names_value]
                fig = px.bar(pd.DataFrame(result.items(), columns=['emoji', 'counts']), y='emoji', x='counts', orientation='h')
                fig.update_layout(yaxis=dict(autorange="reversed"))
    return fig, 'hola'


@app.callback(
    Output("graph", "figure"),
    Input("names", "value"))
def generate_chart(value):
    result = GLOBAL_DF[GLOBAL_DF['file']==value]['data'].values[0]['user_contribution']
    fig = px.pie(pd.DataFrame(result.items(), columns=['user', 'counts']), names='user',values='counts', hole=.3)
    return fig

if __name__ == "__main__":
    WORDCLOUD_DIR.mkdir(exist_ok=True)
    USERNAME = 'Alfonso Ponce'
    cols = ['file', 'data']
    rows = [files for files in data_dir.glob('*.txt')]

    def_rows = []
    all_data = {}
    n = 0
    for r in rows:
        print(f'Procesando {r.name}...')
        if n < 3:

            chat_data = {}
            path = Path(f'Capturas_23052023/{r.name}')
            with open(str(path), 'r', encoding='utf-8') as f:
                df_chat = DS.rawToDf(f.read(), '24hr')
                chat_data['most_used_emojis'] = PF.topKMostUsedEmojis(df_chat, 5, False)
                chat_data['user_contribution'] = PF.messageContributionToChat(df_chat, False)
                chat_data['messages_per_day'] = PF.messagesAmountPerDay(df_chat)
                chat_data['multimedia_sent'] = PF.multimediaSentPerUser(df_chat)
                chat_data['global_bitrigram'] = PF.kBiTriGramCount(df_chat, stopwords, 10)
                store_dir = WORDCLOUD_DIR.joinpath(r.name.replace('.txt', ''))
                store_dir.mkdir(exist_ok=True)
                if not store_dir.joinpath('GeneralWordcloud.png').exists():
                    PF.createGeneralWordcloud(df_chat, stopwords, store_path=store_dir)

            all_data[r.name.replace('.txt', '')] = chat_data
            def_rows.append(str(r))

        n += 1

    GLOBAL_DF = pd.DataFrame(all_data.items(), columns=cols)

    app.run_server(debug=True, use_reloader=False)