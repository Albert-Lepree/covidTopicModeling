from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

def main():
    categories = [
       "talk.religion.misc",
        "talk.politics.misc",
        "sci.med",
        "misc.forsale",
        "rec.sport.baseball" # basketball?
    ]

    pd.set_option('display.max_columns', None)

    df = pd.read_csv('./covid_survey_uk_withdemographics.csv')
    dfGrouped = df.groupby(df.Sex)
    femaleDF = dfGrouped.get_group("Female")
    maleDF = dfGrouped.get_group("Male")

    print(maleDF.text_long)

    # ldaTfidfTopicModeling(femaleDF, "Female Topics from UK Survey Tfidf")
    # ldaTfidfTopicModeling(maleDF, "Male Topics from UK Survey Tfidf")

    countVectorizorLDA(femaleDF, "Female Topics from UK Survey count vectorizer")
    countVectorizorLDA(maleDF, "Male Topics from UK Survey count vectorizer")





def countVectorizorLDA(data, title):
    n_features = 5000

    count_vectorizer = CountVectorizer(lowercase=True,
                                       ngram_range=(1, 1),
                                       max_df=0.95, min_df=2,
                                       max_features=n_features,
                                       stop_words="english"
                                       )

    # Fit and Transform the documents
    X = count_vectorizer.fit_transform(data.text_long)
    # get the actual words from the vectorized data
    feature_names = count_vectorizer.get_feature_names_out()

    n_components = 5
    # Create LDA object
    ldamodel = LatentDirichletAllocation(
        n_components=n_components
    )

    # Fit and Transform model on data that has already been vectorized
    lda_matrix = ldamodel.fit_transform(X)
    # Get Components from the lda model
    # components_[i, j] can be viewed as pseudocount that represents the
    # number of
    # times word j was assigned to topic i.
    lda_components = ldamodel.components_

    plot_top_words(ldamodel, feature_names, 10, title)



def ldaTfidfTopicModeling(data, title):
    n_features = 5000

    tfidf_vectorizer = TfidfVectorizer(lowercase=True,
                                       ngram_range=(1, 1),
                                       max_df=0.95, min_df=2,
                                       max_features=n_features,
                                       stop_words="english"
                                       )

    # Fit and Transform the documents
    X = tfidf_vectorizer.fit_transform(data.text_long)
    # get the actual words from the vectorized data
    feature_names = tfidf_vectorizer.get_feature_names_out()

    n_components = 5
    # Create LDA object
    ldamodel = LatentDirichletAllocation(
        n_components=n_components
    )

    # Fit and Transform model on data that has already been vectorized
    lda_matrix = ldamodel.fit_transform(X)
    # Get Components from the lda model
    # components_[i, j] can be viewed as pseudocount that represents the
    # number of
    # times word j was assigned to topic i.
    lda_components = ldamodel.components_

    plot_top_words(ldamodel, feature_names, 10, title)

def ldaModel(X):
    n_components = 5
    # Create LDA object
    ldamodel = LatentDirichletAllocation(
        n_components=n_components
    )

    # Fit and Transform model on data that has already been vectorized
    lda_matrix = ldamodel.fit_transform(X)
    # Get Components from the lda model
    # components_[i, j] can be viewed as pseudocount that represents the
    # number of
    # times word j was assigned to topic i.
    lda_components = ldamodel.components_


def tfdidf(dataset):
    n_features = 5000

    tfidf_vectorizer = TfidfVectorizer(lowercase=True,
                                       ngram_range=(1, 1),
                                       max_df=0.95, min_df=2,
                                       max_features=n_features,
                                       stop_words="english"
                                       )

    # Fit and Transform the documents
    X = tfidf_vectorizer.fit_transform(dataset)
    # get the actual words from the vectorized data
    tfidf_vectorizer.get_feature_names_out()




import matplotlib.pyplot as plt

def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(1, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[: -n_top_words - 1 : -1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.6)
        ax.set_title(f"Topic {topic_idx +1}", fontdict={"fontsize": 30})
        ax.invert_yaxis()
        ax.tick_params(axis="both", which="major", labelsize=20)
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

