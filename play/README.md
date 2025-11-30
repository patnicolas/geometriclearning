![Banner](../images/GitHub_Banner_Play.png)
### Tutorials

Source code to support Substack newsletter: [Hands-on Geometric Deep Learning](https://patricknicolas.substack.com)     
Patrick Nicolas - Last update 10.28.2025    

![Under](../images/Under_construction.png)


| Tutorial/Play File                  | Substack Article                                                                                                                                                                                                                                    |
|:------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| graph_sage_vs_gcn_play.py           | [Graph Convolutional or SAGE Networks? Shootout](https://patricknicolas.substack.com/p/graph-convolutional-or-sage-networks)                                                                                                                        | 
| graph_to_simplicial_complex_play.py | [Topological Lifting of Graph Neural Networks](https://patricknicolas.substack.com/p/topological-lifting-of-graph-neural)                                                                                                                           |
| featured_simplicial_complex_play.py | [Exploring Simplicial Complexes for Deep Learning: Concepts to Code](https://patricknicolas.substack.com/p/exploring-simplicial-complexes-for)                                                                                                      |
| graph_sage_model_play.py            | [Revisiting Inductive Graph Neural Networks](https://patricknicolas.substack.com/p/revisiting-inductive-graph-neural)                                                                                                                               |     
| son_group_play.py                   | [A Journey into the Lie Group SO(4)](https://patricknicolas.substack.com/p/a-journey-into-the-lie-group-so4)  <br/>  [Mastering Special Orthogonal Groups With Practice](https://patricknicolas.substack.com/p/mastering-special-orthogonal-groups) |    
| cf_statistical_manifold_play.py     | [Geometry of Closed-Form Statistical Manifolds](https://patricknicolas.substack.com/p/geometry-of-closed-form-statistical)                                                                                                                          |
| fisher_rao_play.py                  | [Shape Your Models with the Fisher-Rao Metric](https://patricknicolas.substack.com/p/shape-your-models-with-the-fisher)                                                                                                                             |                                                                                                                                                                                              |
| lie_se3_group_play.py               | [SE3: The Lie Group That Moves the World](https://patricknicolas.substack.com/p/se3-the-lie-group-that-moves-the)                                                                                                                                   |   
| gnn_tuning_play.py                  | [How to Tune a Graph Convolutional Network](https://patricknicolas.substack.com/p/how-to-tune-a-graph-convolutional)                                                                                                                                |           
| gnn_training_play.py                | [Plug & Play Training for Graph Convolutional Networks](https://patricknicolas.substack.com/p/plug-and-play-training-for-graph)                                                                                                                     |
| graph_homophily_play.py             | [Neighbors Matter: How Homophily Shapes Graph Neural Networks](https://patricknicolas.substack.com/p/neighbors-matter-how-homophily-shapes)                                                                                                         |
| graph_data_loader_play.py           | [Demystifying Graph Sampling & Walk Methods](https://patricknicolas.substack.com/p/demystifying-graph-sampling-and-walk)                                                                                                                            |
| gn_memory_monitor_play.py           | [Slimming the Graph Neural Network Footprint](https://patricknicolas.substack.com/p/slimming-the-graph-neural-network)                                                                                                                              |
| featured_cell_complex_play.py | [Graphs Reimagined: The Power of Cell Complexes](https://patricknicolas.substack.com/p/graphs-reimagined-the-power-of-cell)                                                                                                                         |
| featured_hypergraph_play.py | [Exploring Hypergraph with TopoX Library]()                                                                                                                                                                                                         |

## Description   
✅ graph_sage_vs_gcn_play.py     
Comparison of relative performance of GraphSAGE and GCN over PyTorch Geometric datasets. The list evaluation parameters include number of graph SAGE or convolutional layers, scope of neighborhood message aggregation and homophily.   
    
✅ graph_to_simplicial_complex_play.py     
Description, design and implementation of the process of lifting of graphs to simplicial complexes using PyTorch Geometric datasets.    
   
✅ abstract_simplicial_complex_play.py    
Introduction and evaluation of simplicial complexes using TopoNetX library, including computation of incidence matrices and various Laplacians.

✅ graph_sage_play.py      
Description and evaluation of the GraphSAGE graph neural model using PyTorch Geometric     

✅ son_group_play.py      
- Introduction, implementation of SO(4) Lie group with evaluation of generation of random rotation, composition of rotation, inverse rotation and project.    
- Evaluation of SO(2) and SO(3) Lie groups with generation of random rotation, exponential and logarithm maps, compose, inverse and projection operations.      

✅ cf_statistical_manifold_.py    
- Description and evaluation of Riemannian manifolds for the Exponential, Poisson, Binomial and Geometric distributions using closed-form formulas 
  
✅ fisher_rao_play.py    
- Description and evaluation of Fisher-Rao metric with application to closed-form distributions such as exponential, geometric, Poisson and Binomial.   

✅ lie_se3_group_play.py    
- Description and evaluation of Special Euclidean Group of 3 dimension

✅ gnn_tuning_play.py 
- Evaluation of optuna Python library to optimize the hyperparameters for Graph Convolutional Network using PyTorch Geometric Dataset

✅ gnn_training_play.py
- Implementation and evaluation of the training of a Graph Neural Network with application to Graph Convolutional Network using Pytorch Geometric Datasets/

✅ graph_homophily_play.py
- Evaluation of Node and Edge Homophily on a performance of a graph convolutional neural network

✅ graph_data_loader_play.py
- Evaluation of various data loader and neighborhood sampling for message aggregation for a graph convolutional network.

✅ gn_memory_monitor_play.py
- Evaluate various methods to reduce memory usage during training of Graph Neural Networks with PyTorch such as pined memory, 32 and 16-bit float mixed-memory, Activation checkpointing, Optimization of neighbor loader, batch size and training configuration.

✅ featured_cell_simplex_play.py
- Introduction and evaluation of cell complexes using TopoNetX library, including computation of incidence matrices and various Laplacians.

✅ featured_hypergraph_play.py
- Evaluation of hypergraphs using TopoNetX library, including computation of incidence matrices and various Laplacians.


## Source Code Tree
The source tree is organized as follows: 
- Features in __python/__ 
- Unit tests in __tests/__ 
- Newsletter specific evaluation code in __play/__



