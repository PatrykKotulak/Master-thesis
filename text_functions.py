import re
import demoji
import pandas as pd


def remove_quoting_comments(data):
    pattern = r'^RT.*'
    remove = data['Comment'].str.contains(pattern)
    data = data[~remove].reset_index(drop=True)
    return data


def extract_emoticons(text: str,
                      emoji_for_response_0: list,
                      emoji_for_response_1: list,
                      emoji_for_response_2: list
                      ):
    emoji = demoji.findall_list(text[0], desc=False)
    pattern = re.compile(r'[:;Xx]-?[\)\(dD](?=[\s\.]*)')
    text_emoji = pattern.findall(text[0].lower())
    result = [*emoji, *text_emoji]
    if len(result) > 0:
        if text[1] == 0:
            emoji_for_response_0.extend(result)
        elif text[1] == 1:
            emoji_for_response_1.extend(result)
        else:
            emoji_for_response_2.extend(result)
        return ' '.join(set(result)), len(result)
    else:
        return '', 0


def create_dataframe_emoji_comparison(dict_count_emoji_0: dict,
                                      dict_count_emoji_1: dict,
                                      dict_count_emoji_2: dict,
                                      sort_data=['Response_0', 'Response_1']):
    # create dataframe for comparison emoji
    plot_emoticons = pd.concat(
        [pd.DataFrame(dict_count_emoji_0.items(), columns=['emoji',
                                            'Response_0']).set_index('emoji'),
            pd.DataFrame(dict_count_emoji_1.items(), columns=['emoji',
                                            'Response_1']).set_index('emoji'),
            pd.DataFrame(dict_count_emoji_2.items(), columns=['emoji',
                                            'Response_2']).set_index('emoji')],
        axis=1).sort_values(by=sort_data, ascending=False)

    # select most frequency emoji
    plot_emoticons = plot_emoticons[~((plot_emoticons['Response_0'] < 10) &
                                      (plot_emoticons['Response_1'].isna()) &
                                      (plot_emoticons['Response_2'].isna()))]
    return plot_emoticons


def preprocess_text(data: pd.Series):
    # remove of @name
    pattern = re.compile(r'@\w+\s')
    data = data.str.replace(pattern, '')

    # remove of links https
    pattern = re.compile(r"https?[:\/\/]+[a-zA-Z0-9.\-\/?=_~:#%]+")
    data = data.str.replace(pattern, '')

    # removal of punctuations and numbers
    pattern = re.compile(r'[^_ąćęłńóśźżĄĆĘŁŃÓŚŹŻa-zA-Z\s]')
    data = data.str.replace(pattern, '')

    # remove more than one space
    pattern = re.compile(r'\s+')
    data = data.str.replace(pattern, ' ')

    # remove beginning and ending task space
    pattern = re.compile(r'^\s+|\s+?$')
    data = data.str.replace(pattern, '')

    # removal of capitalization
    data = data.str.lower()

    return data
