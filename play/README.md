![Banner](../images/GitHub_Banner_Play.png)
### Tutorials

Source code to support Substack newsletter: [Hands-on Geometric Deep Learning](https://patricknicolas.substack.com)     
Patrick Nicolas - Last update 09.23.2025    

![Under](../images/Under_construction.png)


| Tutorial/Play File                  | Substack Article                                                                                                                                                                                                                                    |
|:------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| graph_sage_vs_gcn_play.py           | [Graph Convolutional or SAGE Networks? Shootout](https://patricknicolas.substack.com/p/graph-convolutional-or-sage-networks)                                                                                                                                                                                                  | 
| graph_to_simplicial_complex_play.py | [Topological Lifting of Graph Neural Networks](https://patricknicolas.substack.com/p/topological-lifting-of-graph-neural)                                                                                                                           |
| abstract_simplicial_complex_play.py | [Exploring Simplicial Complexes for Deep Learning: Concepts to Code](https://patricknicolas.substack.com/p/exploring-simplicial-complexes-for)                                                                                                      |
| graph_sage_model_play.py            | [Revisiting Inductive Graph Neural Networks](https://patricknicolas.substack.com/p/revisiting-inductive-graph-neural)                                                                                                                               |     
| son_group_play.py                   | [A Journey into the Lie Group SO(4)](https://patricknicolas.substack.com/p/a-journey-into-the-lie-group-so4)  <br/>  [Mastering Special Orthogonal Groups With Practice](https://patricknicolas.substack.com/p/mastering-special-orthogonal-groups) |    
| fisher_rao_play.py                  | [Shape Your Models with the Fisher-Rao Metric](https://patricknicolas.substack.com/p/shape-your-models-with-the-fisher)                                                                                                                             |                                                                                                                                                                                              |
| lie_se3_group_play.py               | [SE3: The Lie Group That Moves the World](https://patricknicolas.substack.com/p/se3-the-lie-group-that-moves-the)                                                                                                                                   |   
| gnn_tuning_play.py  | [How to Tune a Graph Convolutional Network](https://patricknicolas.substack.com/p/how-to-tune-a-graph-convolutional)                                                                                                                                |           
| gnn_training_play.py | [Plug & Play Training for Graph Convolutional Networks](https://patricknicolas.substack.com/p/plug-and-play-training-for-graph)                                                                                                                     |

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

✅ fisher_rao.play.py    
- Description and evaluation of Fisher-Rao metric with application to closed-form distributions such as exponential, geometric, Poisson and Binomial.   
  
✅ lie_se3_group_play.py    
- Description and evaluation of Special Euclidean Group of 3 dimension

✅ gnn_tuning_play.py 
- Evaluation of optuna Python library to optimize the hyperparameters for Graph Convolutional Network using PyTorch Geometric Dataset

✅ gnn_training_play.py
- Implementation and evaluation of the training of a Graph Neural Network with application to Graph Convolutional Network using Pytorch Geometric Datasets/

## Source Code Tree
The source tree is organized as follows: 
- Features in __python/__ 
- Unit tests in __tests/__ 
- Newsletter specific evaluation code in __play/__



