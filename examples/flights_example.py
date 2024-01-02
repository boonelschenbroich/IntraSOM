# IntraSOM Library
import intrasom

# Results Clustering and Plotting Modules
from intrasom.visualization import PlotFactory
from intrasom.clustering import ClusterFactory

# Other importations
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt


def train(name):
    data = pd.read_excel("data/{}.xlsx".format(name), index_col=0)

    columns_to_drop = get_columns_to_drop(data)

    data = data.drop(columns=columns_to_drop)

    map_size = (data.shape[0], data.shape[1])

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


def get_columns_to_drop(data):
    columns_to_drop = []
    for column in data.columns:
        series = data[column]
        sum = series.sum()

        if sum < 20:
            columns_to_drop.append(column)
        ##
    ##
    return columns_to_drop


def main():
    name = "Flights_200"

    train(name)

    data = pd.read_excel("data/{}.xlsx".format(name), index_col=0)
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

    # print(som_sample.imput_missing())
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
                      file_name="umatrix_flight_sample_labels",
                      file_path=False,
                      watermark_neurons=False,
                      samples_label=True,
                      samples_label_index=range(18),
                      samples_label_fontsize=8,
                      save_labels_rep=True)

    # data_proj = pd.read_excel("data/Animais_proj.xlsx", index_col=0)
    # print(data_proj)
    #
    # proj_data_result = som_sample.project_nan_data(data_proj=data_proj)
    # print(proj_data_result)
    #
    # rep_dic = som_sample.rep_sample(save=True, project=proj_data_result)
    # sorted_dict = dict(sorted(rep_dic.items()))
    # print(sorted_dict)
    #
    # plot.plot_umatrix(figsize=(13, 2.5),
    #                   hits=True,
    #                   title="U-Matrix - Labeled Representative Samples",
    #                   title_size=20,
    #                   title_pad=20,
    #                   legend_title="Distance",
    #                   legend_title_size=12,
    #                   legend_ticks_size=7,
    #                   label_title_xy=(0, 0.5),
    #                   save=True,
    #                   file_name="umatrix_projected_data",
    #                   file_path=False,
    #                   watermark_neurons=False,
    #                   project_samples_label=proj_data_result,
    #                   samples_label=True,
    #                   samples_label_index=range(19),
    #                   samples_label_fontsize=8,
    #                   save_labels_rep=True)


if __name__ == "__main__":
    main()
