import unittest
import logging
import os
from typing import Dict, Any
import networkx as nx
import toponetx as tnx
from dataset.graph.pyg_datasets import PyGDatasets
from topology.simplicial.graph_to_simplicial_complex import GraphToSimplicialComplex
from python import SKIP_REASON


def lift_from_graph_cliques(graph: nx.Graph, params: Dict[str, Any]) -> tnx.SimplicialComplex:
    from toponetx.transform import graph_to_clique_complex

    logging.info('Graph lifted from NetworkX cliques with max rank 2')
    return graph_to_clique_complex(graph, max_rank=params.get('max_rank', 2))


def lift_from_graph_neighbors(graph: nx.Graph, params: Dict[str, Any]) -> tnx.SimplicialComplex:
    from toponetx.transform import graph_to_neighbor_complex

    logging.info('Graph lifted from node neighbors')
    return graph_to_neighbor_complex(graph)


def lift_from_graph_vietoris_rips(graph: nx.Graph, params: Dict[str, Any]) -> tnx.SimplicialComplex:
    from toponetx.transform import weighted_graph_to_vietoris_rips_complex
    r = 2
    max_dim = 3
    logging.info(f'Graph lifted from Vietoris Rips Complex with radius {r} and max_dim {max_dim}')
    return weighted_graph_to_vietoris_rips_complex(graph, r=r, max_dim=max_dim)


class GraphToSimplicialComplexTest(unittest.TestCase):

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_cora_lift_cliques(self):
        graph_to_simplicial = GraphToSimplicialComplex(dataset='Cora',
                                                       nx_graph=None,
                                                       lifting_method=lift_from_graph_cliques)
        logging.info(graph_to_simplicial.nx_graph)
        self.assertEqual(graph_to_simplicial.nx_graph.number_of_nodes(), 2708)
        self.assertEqual(graph_to_simplicial.nx_graph.number_of_edges(), 5278)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_cora_lift_neighbors(self):
        graph_to_simplicial = GraphToSimplicialComplex(dataset='Cora',
                                                       nx_graph=None,
                                                       lifting_method=lift_from_graph_neighbors)
        logging.info(graph_to_simplicial.nx_graph)
        self.assertEqual(graph_to_simplicial.nx_graph.number_of_nodes(), 2708)
        self.assertEqual(graph_to_simplicial.nx_graph.number_of_edges(), 5278)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_cora_lift_VR(self):
        pyg_dataset = PyGDatasets('Cora')
        graph_to_simplicial = GraphToSimplicialComplex(dataset=pyg_dataset(),
                                                       nx_graph=None,
                                                       lifting_method=lift_from_graph_vietoris_rips)
        logging.info(graph_to_simplicial.nx_graph)
        self.assertEqual(graph_to_simplicial.nx_graph.number_of_nodes(), 2708)
        self.assertEqual(graph_to_simplicial.nx_graph.number_of_edges(), 5278)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_flickr_graph(self):
        graph_to_simplicial = GraphToSimplicialComplex(dataset='Flickr',
                                                       nx_graph=None,
                                                       lifting_method=lift_from_graph_cliques)
        logging.info(graph_to_simplicial.nx_graph)
        self.assertEqual(graph_to_simplicial.nx_graph.number_of_nodes(), 89250)
        self.assertEqual(graph_to_simplicial.nx_graph.number_of_edges(), 449878)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_add_faces_cliques(self):
        import time

        dataset_name = 'Cora'
        graph_to_simplicial = GraphToSimplicialComplex(dataset=dataset_name,
                                                       nx_graph=None,
                                                       lifting_method=lift_from_graph_cliques)
        start = time.time()
        tnx_simplicial = graph_to_simplicial.add_faces({'max_rank': 2})
        logging.info(f'Duration lift_from_graph_cliques for {dataset_name} {time.time() - start}')
        logging.info(tnx_simplicial)

        shape = tnx_simplicial.shape
        self.assertEqual(shape, (2708, 5278, 1630))

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_add_faces_neighbors(self):
        import time
        dataset_name = 'KarateClub'
        graph_to_simplicial = GraphToSimplicialComplex(dataset=dataset_name,
                                                       nx_graph=None,
                                                       lifting_method=lift_from_graph_neighbors)
        start = time.time()
        tnx_simplicial = graph_to_simplicial.add_faces()
        logging.info(f'Duration lift_from_graph_neighbors for {dataset_name} {time.time() - start}')
        logging.info(tnx_simplicial)
        shape = tnx_simplicial.shape

        self.assertEqual(len(shape), 18)
        self.assertEqual(shape[0], 34)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_add_faces_vietoris_rips(self):
        graph_to_simplicial = GraphToSimplicialComplex(dataset='KarateClub',
                                                       nx_graph=None,
                                                       lifting_method=lift_from_graph_vietoris_rips)
        tnx_simplicial = graph_to_simplicial.add_faces()
        logging.info(tnx_simplicial)
        shape = tnx_simplicial.shape
        self.assertEqual(shape, (34, 78, 45, 11))

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_pubmed_count(self):
        import time
        graph_to_simplicial = GraphToSimplicialComplex(dataset='PubMed',
                                                       nx_graph=None,
                                                       lifting_method=lift_from_graph_cliques)
        start = time.time()
        tnx_simplicial = graph_to_simplicial.add_faces({'max_rank': 2})
        logging.info(f'Duration lift_from_graph_cliques {time.time() - start}')
        counts = GraphToSimplicialComplex.count_simplex_by_type(tnx_simplicial)
        logging.info(counts)
        self.assertEqual(counts['triangles'], 12520)
        self.assertEqual(counts['edges'], 44324)

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_init_pubmed_latency_neighbors(self):
        import time
        graph_to_simplicial = GraphToSimplicialComplex(dataset='PubMed',
                                                       nx_graph=None,
                                                       lifting_method=lift_from_graph_neighbors)
        start = time.time()
        tnx_simplicial = graph_to_simplicial.add_faces()
        logging.info(f'Duration lift_from_graph_neighbors {time.time() - start}')

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_features_from_hodge_laplacian_pub_med(self):
        import time

        dataset_name = 'KarateClub'
        # Step 1: Configure the migration from Graph to Simplicial
        start = time.time()
        graph_to_simplicial = GraphToSimplicialComplex(dataset=dataset_name,
                                                       nx_graph=None,
                                                       lifting_method=lift_from_graph_cliques)
        # Step 2: Add faces to existing graph nodes and edges
        tnx_simplicial = graph_to_simplicial.add_faces({'max_rank': 2})

        # Step 3: Generate the simplicial elements for nodes, edges and faces.
        #         Number of eigenvectors for node is 4, edges 5 and faces 4
        num_eigenvectors = (10, 9, 9)
        node_simplicial_elements, edge_simplicial_elements, face_simplicial_elements = (
            GraphToSimplicialComplex.features_from_hodge_laplacian(tnx_simplicial, num_eigenvectors)
        )
        logging.info(f"{dataset_name}: {len(node_simplicial_elements)} nodes,  {len(edge_simplicial_elements)} edges, "
                     f"{len(face_simplicial_elements)} faces")
        nodes_elements = [node for idx, node, in enumerate(node_simplicial_elements) if idx < 3]
        edges_elements = [edge for idx, edge, in enumerate(edge_simplicial_elements) if idx < 3]
        faces_elements = [face for idx, face, in enumerate(face_simplicial_elements) if idx < 3]
        nodes_elements_str = '\n'.join([str(s) for s in nodes_elements])
        edges_elements_str = '\n'.join([str(s) for s in edges_elements])
        faces_elements_str = '\n'.join([str(s) for s in faces_elements])
        logging.info(f'Duration: {time.time() - start}')
        logging.info(f"\nNodes: {num_eigenvectors[0]}, Edges: {num_eigenvectors[1]}, Faces: {num_eigenvectors[2]} "
                     f"eigenvectors\nSimplicial nodes:\n{nodes_elements_str}\nSimplicial edges:\n{edges_elements_str}"
                     f"\nSimplicial faces:\n{faces_elements_str}")

    @unittest.skipIf(os.getenv('SKIP_TESTS_IN_PROGRESS', '0') == '1', reason=SKIP_REASON)
    def test_features_from_hodge_laplacian_karate_club(self):
        graph_to_simplicial = GraphToSimplicialComplex(dataset='KarateClub',
                                                       nx_graph=None,
                                                       lifting_method=lift_from_graph_cliques)
        tnx_simplicial = graph_to_simplicial.add_faces({'max_rank': 2})
        num_eigenvectors = (4, 5, 4)
        _, _, face_simplicial_elements = (
            GraphToSimplicialComplex.features_from_hodge_laplacian(tnx_simplicial, num_eigenvectors)
        )
        result = [face for idx, face, in enumerate(face_simplicial_elements) if idx < 3]
        self.assertEqual(result[0].node_indices,  (0, 1, 2))
        logging.info(result[0])
