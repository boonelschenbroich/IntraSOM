import json
import os
from collections import Counter

import pandas as pd
from pydbt.data_io.lrn_file_reader import read_lrn_file

from intrasom import intrasom
from intrasom.clustering import ClusterFactory
from intrasom.visualization import PlotFactory


def train(name):
    data = load_data()

    map_size = (80, 50)

    som_sample = intrasom.SOMFactory.build(data,
                                           mapsize=map_size,
                                           mapshape='toroid',
                                           lattice='hexa',
                                           normalization='var',
                                           initialization='random',
                                           neighborhood='gaussian',
                                           training='batch',
                                           name=name,
                                           component_names=None,
                                           unit_names=None,
                                           sample_names=None,
                                           missing=False,
                                           save_nan_hist=False,
                                           pred_size=0)
    som_sample.train(train_len_factor=2,
                     previous_epoch=True)

    return som_sample


def load_data():
    file_name = '/Volumes/Data/Projects/PUBonServer/24TCGA/09Originale/24GeneExpressionStage4ExprimiertN286C19821ALU.lrn'
    lrn_file = read_lrn_file(file_name)
    lrn_data = lrn_file.data
    header = lrn_file.header

    colnames = []
    uniques = []
    duplicates = []
    duplicate_ids = []

    for i, v in enumerate(filter(lambda value: value != '' and value != 'Key', header)):
        colnames.append(v)
        if v in uniques:
            if v not in duplicates:
                duplicates.append(v)
                duplicate_ids.append(i)
            ##
        else:
            uniques.append(v)
        ##
    ##


    key = lrn_file.keys
    data = pd.DataFrame(lrn_data, columns=colnames, index=key)
    data.drop(data.columns[duplicate_ids], axis=1, inplace=True)

    return data


def main():
    name = "genes"

    if not os.path.exists("Results/{}_neurons.parquet".format(name)):
        # train if the file is missing
        train(name)
    ##

    data = load_data()

    bmus = pd.read_parquet("Results/{}_neurons.parquet".format(name))
    params = json.load(open("Results/params_{}.json".format(name), encoding='utf-8'))

    som_sample = intrasom.SOMFactory.load_som(data=data,
                                              trained_neurons=bmus,
                                              params=params)

    print(som_sample.results_dataframe)
    print(som_sample.neurons_dataframe)

    rep_dic = som_sample.rep_sample(save=True)
    # Ordering BMUs in ascending order
    sorted_dict = dict(sorted(rep_dic.items()))
    print(sorted_dict)

    #
    plot = PlotFactory(som_sample)

    plot.plot_umatrix(figsize=(13, 5),
                      hits=True,
                      title="U-Matrix - Labeled Representative Data",
                      title_size=20,
                      title_pad=20,
                      legend_title="Distance",
                      legend_title_size=12,
                      legend_ticks_size=7,
                      label_title_xy=(0, 0.5),
                      save=True,
                      file_name="umatrix_gene_sample_labels",
                      file_path=False,
                      watermark_neurons=False,
                      samples_label=True,
                      samples_label_index=range(data.shape[0]),
                      samples_label_fontsize=8,
                      save_labels_rep=True)

    clustering = ClusterFactory(som_sample)
    clusters = clustering.kmeans(k=2)
    clustering.results_cluster(clusters, savetype='parquet')

    clustering.plot_kmeans(figsize=(12, 5),
                           clusters=clusters,
                           title_size=18,
                           title_pad=20,
                           umatrix=True,
                           colormap="gist_rainbow",
                           alfa_clust=0.5,
                           hits=True,
                           legend_text_size=7,
                           cluster_outline=True,
                           plot_labels=True,
                           clusterout_maxtext_size=12,
                           save=True,
                           file_name="cluster_gene_sample_merge")

    proj_data_result = som_sample.project_nan_data(data_proj=data, save=False)
    print(proj_data_result)

    rep_dic = som_sample.rep_sample(save=True, project=proj_data_result)
    sorted_dict = dict(sorted(rep_dic.items()))
    print(sorted_dict)

    plot.plot_umatrix(figsize=(13, 2.5),
                      hits=True,
                      title="U-Matrix - Labeled Representative Samples",
                      title_size=20,
                      title_pad=20,
                      legend_title="Distance",
                      legend_title_size=12,
                      legend_ticks_size=7,
                      label_title_xy=(0, 0.5),
                      save=True,
                      file_name="umatrix_gene_sample_projected_data",
                      file_path=False,
                      watermark_neurons=False,
                      project_samples_label=proj_data_result,
                      samples_label=True,
                      samples_label_index=range(data.shape[0]),
                      samples_label_fontsize=8,
                      save_labels_rep=True)


if __name__ == "__main__":
    main()
