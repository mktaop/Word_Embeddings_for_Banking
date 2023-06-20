#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 14:11:02 2023

@author: avi_patel
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import re
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

data=pd.read_csv('/Users/avi_patel//Downloads/complaints-2023-02-26_13_52.csv') #from cfpb.com
data.info()

ndata=pd.DataFrame(data['Consumer complaint narrative'])
ndata=ndata.rename(columns={ndata.columns[0]:'comment'})
ndata.comment=ndata.comment.astype(str)
ndata.comment=ndata.comment.str.lower()
stop_words=stopwords.words('english')
new_stopwords=["xxxx","xxxxxxxx"]
stop_words.extend(new_stopwords)
#ndata.comment=ndata['comment'].apply(lambda x:''.join(
    #[word for word in x.split() if word not in (stop_words)]))
ndata2=ndata.sample(frac=.01, replace=True, random_state=1) # 1% sample to play with
play=ndata.tail(3)

for i in range(len(play)):
    text=' '.join(play.comment)
    sent_list=nltk.sent_tokenize(text)
    sent_list = [''.join([char for char in line if char.isalnum() or char == ' ']) for line in sent_list]

vect = CountVectorizer(stop_words=None, token_pattern=r"(?u)\b\w+\b")
X = vect.fit_transform(sent_list)
uniq_wrds = vect.get_feature_names()
n = len(uniq_wrds)
co_mat = np.zeros((n,n))

window_len = 5
def update_co_mat(x):   
    # Get all the words in the sentence and store it in an array wrd_lst
    wrd_list = x.split(' ')
    wrd_list = [ele for ele in wrd_list if ele.strip()]
    #print(wrd_list)
    
    # Consider each word as a focus word
    for focus_wrd_indx, focus_wrd in enumerate(wrd_list):
        focus_wrd = focus_wrd.lower()
        # Get the indices of all the context words for the given focus word
        for contxt_wrd_indx in range((max(0,focus_wrd_indx - window_len)),(min(len(wrd_list),focus_wrd_indx + window_len +1))):                        
            # If context words are in the unique words list
            if wrd_list[contxt_wrd_indx] in uniq_wrds:
                
                # To identify the row number, get the index of the focus_wrd in the uniq_wrds list
                co_mat_row_indx = uniq_wrds.index(focus_wrd)
                
                # To identify the column number, get the index of the context words in the uniq_wrds list
                co_mat_col_indx = uniq_wrds.index(wrd_list[contxt_wrd_indx])
                                
                # Update the respective columns of the corresponding focus word row
                co_mat[co_mat_row_indx][co_mat_col_indx] += 1

for sentence in tqdm(play_list):
    update_co_mat(sentence)
df=pd.DataFrame(co_mat, columns=uniq_wrds, index=uniq_wrds)
df.to_csv('/Users/avi_patel/Documents/project_38K.csv')
dfu=pd.DataFrame(uniq_wrds)
dfu.to_csv('/Users/avi_patel/Documents/project_38K_uniq_wrds.csv')

df=pd.read_csv('/Users/avi_patel/Documents/project_38K.csv', index_col=0)
uniq_wrds=pd.read_csv('/Users/avi_patel/Documents/GitHub/cs224u/Data/final/project_38K_uniq_wrds.csv', index_col=0)
uniq_wrds=uniq_wrds.rename(columns={ uniq_wrds.columns[0]: "word" }) 

#...Calcualte positive PMI ...
def observed_over_expected(df):
    col_totals = df.sum(axis=0)
    total = col_totals.sum()
    row_totals = df.sum(axis=1)
    expected = np.outer(row_totals, col_totals) / total
    oe = df / expected
    return oe

def pmi(df, positive=True):
    df = observed_over_expected(df)
    # Silence distracting warnings about log(0):
    with np.errstate(divide='ignore'):
        df = np.log(df)
    df[np.isinf(df)] = 0.0  # log(0) = 0
    if positive:
        df[df < 0] = 0.0
    return df

wghtd_df=pmi(df)

#....look at neighbors using cosine...
def euclidean(u, v):
    return scipy.spatial.distance.euclidean(u, v)

def cosine(u, v):
    return scipy.spatial.distance.cosine(u, v)

def neighbors(word, df, distfunc=cosine):
    """
    Tool for finding the nearest neighbors of `word` in `df` according
    to `distfunc`. The comparisons are between row vectors.

    Parameters
    ----------
    word : str
        The anchor word. Assumed to be in `rownames`.

    df : pd.DataFrame
        The vector-space model.

    distfunc : function mapping vector pairs to floats (default: `cosine`)
        The measure of distance between vectors. Can also be `euclidean`,
        `matching`, `jaccard`, as well as any other distance measure
        between 1d vectors.

    Raises
    ------
    ValueError
        If word is not in `df.index`.

    Returns
    -------
    pd.Series
        Ordered by closeness to `word`.

    """
    if word not in df.index:
        raise ValueError('{} is not in this VSM'.format(word))
    w = df.loc[word]
    dists = df.apply(lambda x: distfunc(w, x), axis=1)
    return dists.sort_values()

print(neighbors('account', df, distfunc=cosine).head())
print(neighbors('account', wghtd_df, distfunc=cosine).head())


#....Subwords...
def ngram_vsm(df, n=2):
    """Create a character-level VSM from `df`.

    Parameters
    ----------
    df : pd.DataFrame

    n : int
        The n-gram size.

    Returns
    -------
    pd.DataFrame
        This will have the same column dimensionality as `df`, but the
        rows will be expanded with representations giving the sum of
        all the original rows in `df` that contain that row's n-gram.

    """
    unigram2vecs = defaultdict(list)
    for w, x in df.iterrows():
        for c in get_character_ngrams(w, n):
            unigram2vecs[c].append(x)
    unigram2vecs = {c: np.array(x).sum(axis=0)
                    for c, x in unigram2vecs.items()}
    cf = pd.DataFrame(unigram2vecs).T
    cf.columns = df.columns
    return cf

def get_character_ngrams(w, n):
    """Map a word to its character-level n-grams, with boundary
    symbols '<w>' and '</w>'.

    Parameters
    ----------
    w : str

    n : int
        The n-gram size.

    Returns
    -------
    list of str

    """
    if n > 1:
        w = ["<w>"] + list(w) + ["</w>"]
    else:
        w = list(w)
    return ["".join(w[i: i+n]) for i in range(len(w)-n+1)]

def character_level_rep(word, cf, n=4):
    """Get a representation for `word` as the sum of all the
    representations of `n`grams that it contains, according to `cf`.

    Parameters
    ----------
    word : str
        The word to represent.

    cf : pd.DataFrame
        The character-level VSM (e.g, the output of `ngram_vsm`).

    n : int
        The n-gram size.

    Returns
    -------
    np.array

    """
    ngrams = get_character_ngrams(word, n)
    ngrams = [n for n in ngrams if n in cf.index]
    reps = cf.loc[ngrams].values
    return reps.sum(axis=0)

df_ngrams = ngram_vsm(df, n=4)
wghtd_df_ngrams=ngram_vsm(wghtd_df, n=4)
'bananas' in df.index #accentuate
accentuatue = character_level_rep("bananas", wghtd_df_ngrams)
account = character_level_rep("account", wghtd_df_ngrams)
cosine(account, accentuatue)

#...Visualize
def tsne_viz(df, colors=None, output_filename=None, figsize=(40, 50), random_state=None):
    """
    2d plot of `df` using t-SNE, with the points labeled by `df.index`,
    aligned with `colors` (defaults to all black).

    Parameters
    ----------
    df : pd.DataFrame
        The matrix to visualize.

    colors : list of colornames or None (default: None)
        Optional list of colors for the vocab. The color names just
        need to be interpretable by matplotlib. If they are supplied,
        they need to have the same length as `df.index`. If `colors=None`,
        then all the words are displayed in black.

    output_filename : str (default: None)
        If not None, then the output image is written to this location.
        The filename suffix determines the image type. If `None`, then
        `plt.plot()` is called, with the behavior determined by the
        environment.

    figsize : (int, int) (default: (40, 50))
        Default size of the output in display units.

    random_state : int or None
        Optionally set the `random_seed` passed to `PCA` and `TSNE`.

    """
    # Colors:
    vocab = df.index
    if not colors:
        colors = ['black' for i in vocab]
    # Recommended reduction via PCA or similar:
    n_components = 50 if df.shape[1] >= 50 else df.shape[1]
    dimreduce = PCA(n_components=n_components, random_state=random_state)
    X = dimreduce.fit_transform(df)
    # t-SNE:
    tsne = TSNE(n_components=2, random_state=random_state)
    tsnemat = tsne.fit_transform(X)
    # Plot values:
    xvals = tsnemat[: , 0]
    yvals = tsnemat[: , 1]
    # Plotting:
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    ax.plot(xvals, yvals, marker='', linestyle='')
    # Text labels:
    for word, x, y, color in zip(vocab, xvals, yvals, colors):
        try:
            ax.annotate(word, (x, y), fontsize=8, color=color)
        except UnicodeDecodeError:  ## Python 2 won't cooperate!
            pass
    plt.axis('off')
    # Output:
    if output_filename:
        plt.savefig(output_filename, bbox_inches='tight')
    else:
        plt.show()

wghtd_df_250=wghtd_df.sample(frac=.01, replace=True, random_state=1)
tsne_viz(wghtd_df_250, output_filename=None, figsize=(40, 50), random_state=42)

#...Normalize using LSA ....
M_dense=wghtd_df.to_numpy()
M=csr_matrix(M_dense)
lsa = TruncatedSVD(n_components=100, n_iter=7, random_state=42)
M_lsa=lsa.fit_transform(M)
wghtd_df_lsa=pd.DataFrame(M_lsa, index=uniq_wrds)
wghtd_df_lsa.to_csv('/Users/avi_patel/Documents/GitHub/cs224u/Data/final/embd_38K_lsa_100d.csv')
print(neighbors('account', df, distfunc=cosine).head(7))
print(neighbors('account', wghtd_df, distfunc=cosine).head(7))
print(neighbors('account', wghtd_df_lsa, distfunc=cosine).head(7))

#...Smooth using autoencoders...
wgthd_df_lsa_ae = TorchAutoencoder(max_iter=1000, hidden_dim=300, eta=0.01).fit(wghtd_df_lsa)
wgthd_df_lsa_ae.to_csv('/Users/avi_patel/Documents/embd_38K_lsa_ae.csv')