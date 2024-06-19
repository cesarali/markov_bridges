import concurrent.futures
import copy
import os
import subprocess as sp
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.linalg import eigvalsh

from markov_bridges.models.metrics.mmd import (
    compute_mmd,
    gaussian,
    gaussian_emd,
    process_tensor,
)

from markov_bridges.utils.plots.graphs_plots import plot_graphs_list

PRINT_TIME = True

"""
g++ -O2 -std=c++11 -o orca_berlin orca_berlin.cpp -static-libstdc++ -static-libgcc
"""

from markov_bridges import orca_path

ORCA_DIR_STANDARD = orca_path

def read_orbit_counts(file_path):
    """
    Reads a file where each line corresponds to a node in a graph.
    Each line contains 15 or 73 space-separated orbit counts.

    :param file_path: Path to the file to be read.
    :return: A list of lists, where each sublist contains the orbit counts for a node.
    """
    orbit_counts = []
    with open(file_path, "r") as file:
        for line in file:
            counts = line.strip().split(" ")

            # Validate the number of orbit counts
            if len(counts) not in [15, 73]:
                raise ValueError(f"Invalid number of orbit counts on line: {line}")

            # Convert counts to integers
            counts = list(map(int, counts))
            orbit_counts.append(counts)

    return np.asarray(orbit_counts)


def degree_worker(G):
    return np.array(nx.degree_histogram(G))


def add_tensor(x, y):
    x, y = process_tensor(x, y)
    return x + y


def degree_stats(graph_ref_list, graph_pred_list, windows=True, orca_dir=None, is_parallel=True):
    """Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)

    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)
    print(len(sample_ref), len(sample_pred))
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing degree mmd: ", elapsed)
    return mmd_dist


###############################################################################


def spectral_worker(G):
    # eigs = nx.laplacian_spectrum(G)
    eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    # from scipy import stats
    # kernel = stats.gaussian_kde(eigs)
    # positions = np.arange(0.0, 2.0, 0.1)
    # spectral_density = kernel(positions)

    # import pdb; pdb.set_trace()
    return spectral_pmf


def spectral_stats(graph_ref_list, graph_pred_list, is_parallel=True):
    """Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_ref_list):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty):
                sample_pred.append(spectral_density)

        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #   for spectral_density in executor.map(spectral_worker, graph_ref_list):
        #     sample_ref.append(spectral_density)
        # with concurrent.futures.ProcessPoolExecutor() as executor:
        #   for spectral_density in executor.map(spectral_worker, graph_pred_list_remove_empty):
        #     sample_pred.append(spectral_density)
    else:
        for i in range(len(graph_ref_list)):
            spectral_temp = spectral_worker(graph_ref_list[i])
            sample_ref.append(spectral_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            spectral_temp = spectral_worker(graph_pred_list_remove_empty[i])
            sample_pred.append(spectral_temp)
    # print(len(sample_ref), len(sample_pred))

    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing degree mmd: ", elapsed)
    return mmd_dist


###############################################################################


def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
    return hist


def clustering_stats(graph_ref_list, graph_pred_list, windows=True, orca_dir=None, KERNEL=gaussian, bins=100, is_parallel=True):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker, [(G, bins) for G in graph_ref_list]):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker, [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(clustering_hist)
    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(nx.clustering(graph_pred_list_remove_empty[i]).values())
            hist, _ = np.histogram(clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False)
            sample_pred.append(hist)
    try:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd, sigma=1.0 / 10, distance_scaling=bins)
    except:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=KERNEL, sigma=1.0 / 10)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing clustering mmd: ", elapsed)
    return mmd_dist

# maps motif/orbit name string to its corresponding list of indices from orca_berlin output
motif_to_indices = {
    "3path": [1, 2],
    "4cycle": [8],
}

def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for u, v in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph, windows=True, orca_dir=None):
    ORCA_DIR = Path(orca_dir)

    tmp_input_path = ORCA_DIR / "tmp.txt"
    f = open(tmp_input_path, "w")
    f.write(str(graph.number_of_nodes()) + " " + str(graph.number_of_edges()) + "\n")
    for u, v in edge_list_reindexed(graph):
        f.write(str(u) + " " + str(v) + "\n")
    f.close()

    if windows:
        command = "orca.exe  4 ./tmp.txt tmp.out"
        result = sp.run(command, shell=True, cwd=ORCA_DIR, stdout=sp.PIPE, stderr=sp.PIPE)
    else:
        command = "./orca  4 ./tmp.txt tmp.out"
        result = sp.run(command, shell=True, cwd=ORCA_DIR, stdout=sp.PIPE, stderr=sp.PIPE)
        # result = sp.check_output([os.path.join(ORCA_DIR, 'orca'), '4', tmp_input_path, 'tmp.out'])

    tmp_output_file = ORCA_DIR / "tmp.out"
    node_orbit_counts = read_orbit_counts(tmp_output_file)

    try:
        os.remove(tmp_input_path)
        os.remove(tmp_output_file)
    except OSError:
        pass

    return node_orbit_counts


def orbit_stats_all(graph_ref_list, graph_pred_list, windows=True, orca_dir=None):
    total_counts_ref = []
    total_counts_pred = []

    # graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    for G in graph_ref_list:
        try:
            orbit_counts = orca(G, windows, orca_dir=orca_dir)
        except Exception as e:
            print(e)
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    for G in graph_pred_list:
        try:
            orbit_counts = orca(G, windows, orca_dir=orca_dir)
        except Exception as e:
            print(e)
            continue
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)
    mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=gaussian, is_hist=False, sigma=30.0)

    # print('-------------------------')
    # print(np.sum(total_counts_ref, axis=0) / len(total_counts_ref))
    # print('...')
    # print(np.sum(total_counts_pred, axis=0) / len(total_counts_pred))
    # print('-------------------------')
    return mmd_dist


def adjs_to_graphs(adjs, node_flags=None):
    graph_list = []
    for adj in adjs:
        G = nx.from_numpy_matrix(adj)
        G.remove_edges_from(G.selfloop_edges())
        G.remove_nodes_from(list(nx.isolates(G)))
        if G.number_of_nodes() < 1:
            G.add_node(1)
        graph_list.append(G)
    return graph_list


def eval_acc_lobster_graph(G_list):
    G_list = [copy.deepcopy(gg) for gg in G_list]

    count = 0
    for gg in G_list:
        if is_lobster_graph(gg):
            count += 1

    return count / float(len(G_list))


def is_lobster_graph(G):
    """
    Check a given graph is a lobster graph or not

    Removing leaf nodes twice:

    lobster -> caterpillar -> path

    """
    ### Check if G is a tree
    if nx.is_tree(G):
        # import pdb; pdb.set_trace()
        ### Check if G is a path after removing leaves twice
        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)

        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)

        num_nodes = len(G.nodes())
        num_degree_one = [d for n, d in G.degree() if d == 1]
        num_degree_two = [d for n, d in G.degree() if d == 2]

        if sum(num_degree_one) == 2 and sum(num_degree_two) == 2 * (num_nodes - 2):
            return True
        elif sum(num_degree_one) == 0 and sum(num_degree_two) == 0:
            return True
        else:
            return False
    else:
        return False


METHOD_NAME_TO_FUNC = {
    "orbit": orbit_stats_all,
    "degree": degree_stats,
    "cluster": clustering_stats,
    #'spectral': spectral_stats
}


def eval_torch_batch(ref_batch, pred_batch, methods=None):
    graph_ref_list = adjs_to_graphs(ref_batch.detach().cpu().numpy())
    grad_pred_list = adjs_to_graphs(pred_batch.detach().cpu().numpy())
    results = eval_graph_list(graph_ref_list, grad_pred_list, methods=methods)
    return results


def eval_graph_list(graph_ref_list, grad_pred_list, methods=None, windows=True, orca_dir=ORCA_DIR_STANDARD):
    if methods is None:
        # methods = ["orbit"]
        methods = ["degree", "cluster", "orbit"]
    results = {}
    for method in methods:
        try:
            results[method] = METHOD_NAME_TO_FUNC[method](graph_ref_list, grad_pred_list, windows, orca_dir)
        except Exception as e:
            print(e)
            continue
        print(results)
    return results

from markov_bridges.models.metrics.abstract_metrics import BasicMetric
from markov_bridges.configs.config_classes.metrics.metrics_configs import GraphMetricsConfig
from markov_bridges.models.generative_models.cjb import CJB
from markov_bridges.data.abstract_dataloader import MarkovBridgeDataNameTuple
from markov_bridges.models.pipelines.pipeline_cjb import CJBPipelineOutput

##################################################################################################

class GraphsMetrics(BasicMetric):
    """
    """
    def __init__(self, model: CJB, metrics_config: GraphMetricsConfig):
        super().__init__(model, metrics_config)
        self.transform_to_native_shape = model.dataloader.transform_to_native_shape
        self.networkx_from_sample = model.dataloader.networkx_from_sample
        self.plot_graphs = metrics_config.plot_graphs
        self.methods = metrics_config.methods
        self.windows = metrics_config.windows

    def batch_operation(self, databatch: MarkovBridgeDataNameTuple, generative_sample: CJBPipelineOutput):
        pass

    def final_operation(self, all_metrics,samples_gather,epoch=None):
        generative_sample = self.transform_to_native_shape(samples_gather.raw_sample)
        target_discrete = self.transform_to_native_shape(samples_gather.target_discrete)
        
        generative_graphs = self.networkx_from_sample(generative_sample)
        target_graphs = self.networkx_from_sample(target_discrete)

        if self.plot_graphs:
            if self.plots_path is not None:
                plots_path_generative = self.plots_path.format(self.name + "_generative_{0}_".format(epoch))
                plots_path_original = self.plots_path.format(self.name + "_original_{0}_".format(epoch))
            else:
                plots_path_generative = None
                plots_path_original = None
            plot_graphs_list(generative_graphs,title="Generative",save_dir=plots_path_generative)
            plot_graphs_list(target_graphs,title="Original",save_dir=plots_path_original)

        all_metrics = eval_graph_list(target_graphs, generative_graphs, methods=self.methods, windows=self.windows)
        return all_metrics


if __name__=="__main__":
    import subprocess
    from pprint import pprint

    graph = nx.barabasi_albert_graph(200, 3)
    graph_list_1 = [nx.barabasi_albert_graph(200,3) for i in range(10)]
    graph_list_2 = [nx.barabasi_albert_graph(200,3) for i in range(10)]

    #node_orbit_counts = orca(graph_list_1[0])
    results_ = eval_graph_list(graph_list_1, graph_list_2,methods=["orbit"],windows=True,orca_dir=orca_path)
    print(results_)