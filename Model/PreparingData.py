from pathlib import Path
import DataStructure as DS
import Cleaning as C
import pandas as pd
#from whatstk import df_from_txt_whatsapp
pd.options.mode.chained_assignment = None

data_dir = Path('../Capturas_26082023')
root_store_dir = Path('PreparedDataV3')
root_store_dir.mkdir(exist_ok=True)

#stopwords = open('WhatsApp_StopWords.txt', 'r', encoding='utf-8').read().split()
user = 'Alfonso Ponce'
w = 0
objective_data = []
p = False
for files in data_dir.glob('*.txt'):
    with open(str(files), 'r', encoding='utf-8') as f:
        store_dir = root_store_dir.joinpath(files.name.replace('txt','csv'))
        df_chat = DS.rawToDf(f.read(), '24hr')
        #df_chat = df_from_txt_whatsapp(str(files))
        df_chat = C.removeMultimedia(df_chat, is_from_whatstk=False)
        df_chat = C.removeErasedMessages(df_chat)
        df_chat = C.emoji2text(df_chat)
        df_chat = C.removeURL(df_chat)

        print(store_dir.name)
        #df_chat = DS.dfToTrainGPT2(df_chat, user)
        df_chat = DS.dfToTrain(df_chat, user)
        #if df_chat != None:
        df_chat.to_csv(store_dir, index=False)

