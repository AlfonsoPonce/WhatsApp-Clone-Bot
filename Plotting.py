import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as mfm
import matplotlib.font_manager as fm
from matplotlib import ft2font
from matplotlib.font_manager import ttfFontProperty
def pieChart(dict, title):
    """
    Plots pie chart given dict computed from text preprocessing
    :param dict: dict with percentages as values
    :return:
    """

    labels = list(dict.keys())
    sizes = list(dict.values())

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels,autopct='%1.1f%%', radius=1)
    plt.title(title)
    plt.show()

def barPlotNestedDict(dict, title):
    """
    Plots bar plot given dict computed from text preprocessing
    :param dict: dict with percentages as values
    :return:
    """
    #matplotlib.use('tkAgg')
    fpath = 'AppleColorEmoji.ttc'
    fprop = fm.FontProperties(fname=fpath)
    font = ft2font.FT2Font(fpath)
    fprop = fm.FontProperties(fname=fpath)
    ttfFontProp = ttfFontProperty(font)


    fontprop = fm.FontProperties(family='sans-serif',
                                fname=ttfFontProp.fname,
                                size=5,
                                stretch=ttfFontProp.stretch,
                                style=ttfFontProp.style,
                                variant=ttfFontProp.variant,
                                weight=ttfFontProp.weight)
    for user in dict.keys():
        keys = list(dict[user].keys())
        values = list(dict[user].values())
        axis = plt.barh(keys, values)
        axis.set_yticklabels(keys, rotation=0, fontsize=5, fontproperties=fprop)
        plt.title(f'{title} -- {user}')
        plt.show()



