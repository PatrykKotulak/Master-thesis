import pandas as pd
import matplotlib.pyplot as plt
import itertools

from wordcloud import WordCloud


def plot_word_cloud(data: pd.Series, stopwords=[]):
    all_words = ' '.join([text for text in data['Comment']])
    wordcloud = WordCloud(stopwords=stopwords, width=800, height=500,
                          random_state=21,
                          max_font_size=110,
                          background_color="white").generate(all_words)
    # random=0.30
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()


def count_plot_words(data: pd.Series, kind_comment: str):
    words_counts = {}
    for comments in data['Comment']:
        for word in comments.split():
            if word not in words_counts:
                words_counts[word] = 1
            words_counts[word] += 1
    words_counts = sorted(words_counts.items(), key=lambda x: x[1],
                          reverse=True)
    words_counts = dict(words_counts)
    words_counts = dict(itertools.islice(words_counts.items(), 20))

    dataframe_to_plot = pd.DataFrame(list(words_counts.items()),
                                     columns=['Word', 'Number'])
    words = dataframe_to_plot['Word']
    number = dataframe_to_plot['Number']

    # bar plot for counted words - TOP 20

    # Figure Size
    fig, ax = plt.subplots(figsize=(16, 9))

    # Horizontal Bar Plot
    ax.barh(words, number)

    # Remove x, y Ticks
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

    # Add padding between axes and labels
    ax.xaxis.set_tick_params(pad=5)
    ax.yaxis.set_tick_params(pad=10)

    # Add x, y gridlines
    ax.grid(b=True, color='grey',
            linestyle='-.', linewidth=0.5,
            alpha=0.2)

    # Show top values
    ax.invert_yaxis()

    # Add annotation to bars
    for i in ax.patches:
        plt.text(i.get_width() + 0.2, i.get_y() + 0.5,
                 str(round((i.get_width()), 2)),
                 fontsize=10, fontweight='bold',
                 color='grey')

    # Show Plot
    plt.title(f'TOP 20 wystąpień słów ({kind_comment})', fontsize=18)
    plt.xlabel("Liczba wystąpień", fontsize=16)
    plt.ylabel("Słowa", fontsize=16)
    plt.show()
