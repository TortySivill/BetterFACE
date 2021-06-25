import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
import networkx as nx
from typing import Union
import logging
import importlib as imp
imp.reload(logging)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO, datefmt='%I:%M:%S')


class BaseFACE:
    """Base class for CE methods based on shortest path over a graph

    Attributes:
        data: pandas DataFrame containing data (n_samples, n_features)
        clf: classifier with a predict method and predict_proba if probabilities wanted
        pred_threshold: prediction threshold for classification of CE
        dist_threshold: maximum distance between nodes for edge to be added
    """

    def __init__(
            self,
            data: pd.DataFrame,
            clf,
            pred_threshold: float = None,
            bidirectional: bool = False,
            dist_threshold: float = 1
    ):
        self.data = data
        self.clf = clf
        self.prediction = pd.DataFrame(self.clf.predict(data).astype(int), columns=["prediction"], index=data.index)
        self.pred_threshold = pred_threshold
        self.bidirectional = bidirectional
        self.dist_threshold = dist_threshold
        if bidirectional is False:
            self.G = nx.Graph()
        else:
            self.G = nx.DiGraph()
        self.add_nodes_and_edges()
        self.connected_nodes = None

    def _threshold_function(
            self,
            XA: pd.DataFrame,
            XB: pd.DataFrame
    ) -> np.ndarray:
        """Function that determines whether to add edge to graph

        Args:
            XA: Pandas DataFrame containing (n_samples, n_features)
            XB: Pandas DataFrame containing (n_samples, n_features)

        Returns:
            binary matrix of size len(XA) * len(XB)
        """
        return (cdist(XA.values, XB.values, metric='euclidean') < self.dist_threshold).astype(bool)

    def _weight_function(
            self,
            XA: pd.DataFrame,
            XB: pd.DataFrame,
            threshold_matrix: np.ndarray
    ) -> np.ndarray:
        """Distance or density function that calculates weights for graph

        Default uses distance measure for weights but can be overridden in subclasses.

        Args:
            XA: Pandas DataFrame containing (n_samples, n_features)
            XB: Pandas DataFrame containing (n_samples, n_features)
            threshold_matrix: binary matrix of size len(XA) * len(XB)

        Returns:
            weight between node_collection
        """
        with np.errstate(divide='ignore'):
            return cdist(XA.values, XB.values, metric='euclidean') / threshold_matrix

    def prune_nodes(self):
        """Method to remove nodes that do not meet a condition or threshold

        Returns:

        """
        unconnected_nodes = [node for node, deg in self.G.degree if deg == 0]
        self.G.remove_nodes_from(unconnected_nodes)
        if len(unconnected_nodes) > 0:
            logging.info(f' {len(unconnected_nodes)} nodes removed as unconnected. Graph now has {len(self.G.nodes)} nodes.')

    def prune_edges(self):
        """Method to remove edges that do not meet a threshold

        Returns:

        """
        pass

    def add_nodes_and_edges(
            self,
            new_node: int = None
    ):
        """Creates nodes and edges with weights.

        If new_point is False then creates nodes and edges (if threshold is met) for all data node_collection.
        If now_point is True then creates 1 extra node and adds edges to all other nodes.

        Args:
            new_node: boolean whether a new point has just been added to data

        Returns:

        """
        if new_node is None:
            self.G.add_nodes_from(list(self.data.index))
            logging.info(f' Graph has been created with {self.G.number_of_nodes()} nodes.')

        else:
            self.G.add_node(new_node)
            logging.info(f' 1 node has been added to graph. Graph now has {len(self.G.nodes())} nodes.')

        XA = self.data.loc[list(self.G.nodes)]
        if new_node is None: XB = XA
        else: XB = pd.DataFrame(XA.iloc[-1]).T

        node_nums_A, node_nums_B = list(XA.index), list(XB.index)

        threshold_matrix = self._threshold_function(XA, XB)
        weight_matrix = self._weight_function(XA, XB, threshold_matrix)
        edge_weights = [(node_nums_A[a], node_nums_B[b], weight_matrix[a,b])
                        for a in range(weight_matrix.shape[0]) for b in range(weight_matrix.shape[1])
                        if not(np.isinf(weight_matrix[a,b]))]

        self.G.add_weighted_edges_from(edge_weights)
        logging.info(f' {len(edge_weights)} edges have been added to graph.')
        self.prune_edges()
        self.prune_nodes()

    def generate_counterfactual(
            self,
            instance: Union[int, np.ndarray],
            target_class: int = None
    ) -> (pd.DataFrame, pd.DataFrame):
        """Generates counterfactual to flip prediction of example using dijkstra shortest path for the graph created

        Args:
            instance: instance to generate CE, specified either as an int (index in existing data array) or np.array (new data point)
            target_class: target class for CE, if none then opposite class for binary classification

        Returns:
            path to CE as instances from data as a pandas DataFrame
            probability of CE (if prediction_prob specified)

        """
        if type(instance) == int:
            assert instance in self.data.index, "Invalid instance index."
            start_node = instance
            pred = self.prediction.iloc[instance]["prediction"]
            instance = self.data.iloc[instance].values
        else:
            start_node = list(self.G.nodes)[-1] + 1
            self.data = self.data.append(pd.Series(instance.squeeze(), index=list(self.data), name=start_node))
            pred = self.clf.predict(instance.reshape(1,-1))[0].astype(int)
            self.prediction = self.prediction.append(pd.Series(pred, index=list(self.prediction), name=start_node))  
            raise NotImplementedError("UDM implementation currently cannot handle new instance addition.")          
            self.add_nodes_and_edges(start_node)

        # assert target_class != pred, "Target class is the same as the current prediction."
        assert start_node in list(self.G.nodes), "Instance does not meet thresholds."

        if target_class is None: target_class = np.logical_not(pred).astype(int) # NOTE: Only works with binary classification!
        
        logging.info(f' Generating counterfactual for instance {instance} using {self.__class__}. Fact prediction = {pred}, foil = {target_class}.')

        # # TODO: node_connected_component does not work for directed graph find work around
        # self.connected_nodes = list(nx.node_connected_component(self.G, start_node))

        # target_nodes = self.data.loc[self.connected_nodes][(self.prediction.loc[self.connected_nodes] == target_class)
        #                                                     .values].index

        target_nodes = self.prediction.index[self.prediction["prediction"] == target_class]

        assert len(target_nodes) > 0, "No target nodes that meet thresholds."
        logging.info(f' {len(target_nodes)} target nodes found.')

        # NOTE: Running single_source_dijkstra once without specifying a target node is usually faster
        # than running it many times with different targets, because it avoids repeated computation.
        costs, paths = nx.single_source_dijkstra(self.G, start_node)
        paths_and_costs = [(paths[n], costs[n]) for n in target_nodes if n in costs]
        logging.info(f" {len(paths_and_costs)} permissible paths found.")
        if len(paths_and_costs) == 0: return None, None

        paths_and_costs.sort(key=lambda x:x[1]) # Sort by ascending cost.

        i = 0
        if self.pred_threshold is not None:
            pred_probs = self.clf.predict_proba(self.data.loc[paths_and_costs[i][0][-1]].values.reshape(1, -1))\
                .squeeze()[target_class]
            while pred_probs[-1] < self.pred_threshold:
                i += 1
                assert i < len(target_nodes), "Prediction threshold not met."
                pred_probs = self.clf.predict_proba(self.data.loc[paths_and_costs[i][0][-1]].values.reshape(1, -1))\
                    .squeeze()[target_class]
                
        best_path_df = self.path_df(paths_and_costs[i][0], include_probs=(self.pred_threshold is not None))

        return paths_and_costs, best_path_df

    def path_df(
        self, 
        path,
        include_probs=False
    ):
        data = self.data.loc[path]
        df = data
        df["prediction"] = self.prediction.loc[path]["prediction"]
        if include_probs:
            # NOTE: What is take_along_axis doing for us here?
            df["probability"] = np.take_along_axis(self.clf.predict_proba(data), df["prediction"].values, axis=0)
        return df

    def plot_path(
            self,
            path: list#[int] # NOTE: This kind of type hinting doesn't work in Python 3.6.9.
    ):
        """Plots the subgraph of nodes that are connected to the instance and shows the path to the CE

        Args:
            path: list of nodes of the shortest path

        Returns:

        """
        fig, ax = plt.subplots(figsize=(12, 12))
        subG = self.G.subgraph(self.connected_nodes)
        pos = nx.drawing.nx_agraph.graphviz_layout(subG, prog="neato")
        nx.draw_networkx(subG, pos=pos, with_labels=False, node_size=200,
                         node_color=[["purple", "y"][self.prediction.loc[node].item()] for node in self.connected_nodes]
                         )
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(subG, pos=pos, edgelist=path_edges, edge_color='r', width=3)
        fig.show()

    def show(
        self,
        paths_and_costs: list = None,
        prune: bool = True,
        edge_labels = False
    ):
        """Plots all nodes and edges in the graph, optionally highlighting a path.

        Args:
            path: Optional list of nodes.

        Returns:

        """
        plt.figure(figsize=(12, 12))
        if paths_and_costs and prune: 
            # Only show nodes that appear on at least one path
            G = self.G.edge_subgraph([(path[i], path[i+1]) for path, _ in paths_and_costs for i in range(len(path)-1)])
        else: G = self.G
        if paths_and_costs and not prune:
            # If paths specified, and haven't already pruned, highlight them in a different colour.
            path_edges = set((path[i], path[i+1]) for path, _ in paths_and_costs for i in range(len(path)-1))
            invalid_edges = path_edges - G.edges
            assert invalid_edges == set(), f"Invalid edges: {invalid_edges}"
        else: path_edges = set()
        # Set edge width according to inverse weight. 
        weights = np.array(list(nx.get_edge_attributes(G, "weight").values()))
        edge_width = 1 * weights.max() / weights
        # Compute layout.    
        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog="neato")
        node_collection = nx.draw_networkx_nodes(G, 
            pos=pos,
            node_color=[["purple", "y"][self.prediction.loc[node].item()] for node in G.nodes], # NOTE: Only works with binary classification.    
        )
        node_collection.set_zorder(3)
        edge_collection = nx.draw_networkx_edges(G, 
            pos=pos,
            edge_color=["r" if e in path_edges else "k" for e in G.edges],
            width=edge_width,
            connectionstyle="arc3,rad=0.1"if self.bidirectional else None, # Curved edges if bidirectional.
        )
        # Bring path edges to the front.
        for e, a in zip(G.edges, edge_collection): 
            if e in path_edges:
                a.set_zorder(2)
        nx.draw_networkx_labels(G,
            pos=pos,
        )
        if edge_labels:
            label_collection = nx.draw_networkx_edge_labels(G,
                pos=pos,
                label_pos=0.4,
                font_size=6,
                edge_labels={(i, j): f"{d['weight']:.2f}" for i, j, d in G.edges(data=True)},
            )
            for l in label_collection.values(): l.set_zorder(4)