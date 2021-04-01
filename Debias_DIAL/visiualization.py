import numpy as np
from typing import List
import matplotlib.pyplot as plt
from numpy.lib.function_base import select
from sklearn.decomposition import PCA
from sklearn import manifold
from matplotlib.backends.backend_pdf import PdfPages

def pca_visualization(X: np.ndarray,
                      race: np.ndarray,
                      sent: np.ndarray,
                    #   classes: List[str],
                      save_path: str):
    """
    Apply PCA visualization for features.
    """
    red_features = PCA(n_components=2, svd_solver="full").fit_transform(X)

    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots()
    # for _class in classes:
    #     if _class == "unseen":
    # ax.scatter(red_features[sent==0  , 0], red_features[sent==0, 1],
    #         label='pos', alpha=0.5, s=20, edgecolors='none', color="blue")
    # ax.scatter(red_features[sent==1  , 0], red_features[sent==1, 1],
    #         label='neg', alpha=0.5, s=20, edgecolors='none', color="red")
    ax.scatter(red_features[race==0  , 0], red_features[race==0, 1],
            label='AAE', alpha=0.5, s=20, edgecolors='none', color="blue")
    ax.scatter(red_features[race==1  , 0], red_features[race==1, 1],
            label='SAE', alpha=0.5, s=20, edgecolors='none', color="red")
    # ax.scatter(red_features[(race==0)*(sent==0)  , 0], red_features[(race==0)*(sent==0), 1],
    #         label='white_pos', alpha=0.5, s=20, edgecolors='none', color="orange")
    # ax.scatter(red_features[(race==0)*(sent==1)  , 0], red_features[(race==0)*(sent==1), 1],
    #         label='white_neg', alpha=0.5, s=20, edgecolors='none', color="red")
    # ax.scatter(red_features[(race==1)*(sent==0)  , 0], red_features[(race==1)*(sent==0), 1],
    #         label='black_pos', alpha=0.5, s=20, edgecolors='none', color="green")
    # ax.scatter(red_features[(race==1)*(sent==1), 0], red_features[(race==1)*(sent==1), 1],
    #         label='black_neg', alpha=0.5, s=20, edgecolors='none', color="blue")
        # else:
        #     ax.scatter(red_features[y == _class, 0], red_features[y == _class, 1],
        #             label=_class, alpha=0.5, s=20, edgecolors='none', zorder=10)
    ax.legend()
    ax.grid(True)

    # plt.savefig(save_path, format="pdf")
    pp = PdfPages(save_path)
    pp.savefig(plt)
    pp.close()

def tSNE(X: np.ndarray,
        race: np.ndarray,
        sent: np.ndarray,
    #   classes: List[str],
        save_path: str):
    # selected = sent==1
    selected = sent>-1
    X = X[selected]
    race = race[selected]
    sent = sent[selected]
    # print(len(X))
    
    tsne = manifold.TSNE(n_components=2, init='pca',perplexity=2, n_iter=1200,early_exaggeration=200, random_state=11)

    red_features = tsne.fit_transform(X) 
    plt.style.use("seaborn-darkgrid")
    fig, ax = plt.subplots()
    # for _class in classes:
    #     if _class == "unseen":
    # ax.scatter(red_features[sent==0  , 0], red_features[sent==0, 1],
    #         label='pos', alpha=0.5, s=20, edgecolors='none', color="blue")
    # ax.scatter(red_features[sent==1  , 0], red_features[sent==0, 1],
    #         label='neg', alpha=0.5, s=20, edgecolors='none', color="red")
    ax.scatter(red_features[race==0  , 0], red_features[race==0, 1],
            label='AAE', alpha=0.5, s=20, edgecolors='none', color="green")
    ax.scatter(red_features[race==1  , 0], red_features[race==1, 1],
            label='SAE', alpha=0.5, s=20, edgecolors='none', color="black")
    # ax.scatter(red_features[(race==0)*(sent==0)  , 0], red_features[(race==0)*(sent==0), 1],
    #         label='white_pos', alpha=0.5, s=20, edgecolors='none', color="black")
    # ax.scatter(red_features[(race==0)*(sent==1)  , 0], red_features[(race==0)*(sent==1), 1],
    #         label='white_neg', alpha=0.5, s=20, edgecolors='none', color="red")
    # ax.scatter(red_features[(race==1)*(sent==0)  , 0], red_features[(race==1)*(sent==0), 1],
    #         label='black_pos', alpha=0.5, s=20, edgecolors='none', color="green")
    # ax.scatter(red_features[(race==1)*(sent==1), 0], red_features[(race==1)*(sent==1), 1],
    #         label='black_neg', alpha=0.5, s=20, edgecolors='none', color="blue")
        # else:
        #     ax.scatter(red_features[y == _class, 0], red_features[y == _class, 1],
        #             label=_class, alpha=0.5, s=20, edgecolors='none', zorder=10)
    ax.legend()
    ax.grid(True)
    plt.savefig(save_path, format="png")
    # pp = PdfPages(save_path)
    # pp.savefig(plt)
    # pp.close()


def main():
    enc_moji = np.load('./output_baselines/1100_0.8_acc/enc_moji_last.npy')
    # enc_moji = np.load('./output_baselines/1100_0.8_acc/enc_moji.npy')
    race = np.load('./output_baselines/1100_0.8_acc/race_label.npy')
    sent = np.load('./output_baselines/1100_0.8_acc/sent_label.npy')
    # pca_visualization(enc_moji, race, sent, './figs/undebias_0.8_pca.png')
    # tSNE(enc_moji, race, sent, './figs/undebias_0.8_tsne_bftanh_pos.png')
    # tSNE(enc_moji, race, sent, './figs/undebias_0.8_tsne_bftanh_neg.png')
    tSNE(enc_moji, race, sent, './figs/undebias_0.8_tsne_bftanh_all.png')

    enc_moji = np.load('./outputs_embed/seed@1-1-1-1-ratio_0.8_0010_CrossEntropyLoss_1.4/enc_moji_last.npy')
    race = np.load('./outputs_embed/seed@1-1-1-1-ratio_0.8_0010_CrossEntropyLoss_1.4/race_label.npy')
    sent = np.load('./outputs_embed/seed@1-1-1-1-ratio_0.8_0010_CrossEntropyLoss_1.4/sent_label.npy')
    # pca_visualization(enc_moji, race, sent, './figs/debias_0.8_pca.png')
    # tSNE(enc_moji, race, sent, './figs/debias_0.8_tsne_bftanh_pos.png')
    # tSNE(enc_moji, race, sent, './figs/debias_0.8_tsne_bftanh_all.pdf')
    # tSNE(enc_moji, race, sent, './figs/debias_0.8_tsne_bftanh_neg.png')
    tSNE(enc_moji, race, sent, './figs/debias_0.8_tsne_bftanh_all.png')
    
if __name__ == "__main__":
    main()