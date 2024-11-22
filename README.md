# Geometric Learning
Classes and methods for Geometric Learning and related topics.

#### Patrick Nicolas - Last update 11.01.2024

![Banner](images/GeometricLearning.png)

# Theory 
## Differential Geometry
__Differential geometry__ offers a solution by enabling data scientists to grasp the true shape and distribution of data.   
     
Differential geometry is a branch of mathematics that uses techniques from calculus, algebra and topology to study the properties of curves, surfaces, and higher-dimensional objects in space. It focuses on concepts such as curvature, angles, and distances, examining how these properties vary as one moves along different paths on a geometric object.  
Differential geometry is crucial in understanding the shapes and structures of objects that can be continuously altered, and it has applications in many fields including physics (I.e., general relativity and quantum mechanics), engineering, computer science, and data exploration and analysis.   
   
__Euclidean space__, created from a collection of maps (or charts) called an atlas, which belongs to Euclidean space. __Differential manifolds__ have a tangent space at each point, consisting of vectors. __Riemannian__ manifolds are a type of differential manifold equipped with a __metric__ to measure __curvature__, __gradient__, and __divergence__.
   
![Manifold](images/Manifold_Tgt_Space.png)

The directory __geometry__ contains the defintion and implementation the various elements of smooth manifolds using __Geomstats__ module.
- __Covariant__ and __contravariant__ vectors and tensors
- __Intrinsic__ and __Extrinsic__ coordinates
- Riemannian metric tensor
- Tangent space, __Exponential__ and __Logarithmic__ maps
- __Levi-Civita__ connection and parallel transport
- __Curvature__ tensor
- Euclidean, __hypersphere__ and __Kendal__ spaces
- Logistic regression and K-Means on hypersphere
- __Frechet mean__
- __Push-forward__ and __pull-back__     

![Paralleltransport](images/ParallelTransport.png)
    

## Lie Groups and Algebras
Lie groups play a crucial role in Geometric Deep Learning by modeling symmetries such as rotation, translation, and scaling. This enables non-linear models to generalize effectively for tasks like object detection and transformations in generative models.    
Lie groups have numerous practical applications in various fields:     
- __Physics__: They describe symmetries in classical mechanics, quantum mechanics, and relativity. 
- __Robotics__: Lie groups model the motion of robots, particularly in the context of rotation and translation (using groups like SO(3) and SE(3)).
- __Control Theory__: Lie groups are used in the analysis and design of control systems, especially in systems with rotational or symmetrical behavior.
- __Computer Vision__: They help in image processing and 3D vision, especially in tasks involving rotations and transformations.
- __Differential Equations__: Lie groups are instrumental in solving differential equations by leveraging symmetry properties.     

![LieGroups](images/Lie_Manifold.png)

    
The directory __Lie__ illustrates the various element of __Special Orthogonal Group__ of 3 dimension (__SO3__) and __Special Euclidean Group__ in 3 dimension (__SE3__) using __Geomstats__ library. 

## Deep Learning Models
The directory __dl__ implements a framework of __reusable neural blocks__ as key components of any deep learning models such as:
- Feed forward neural network
- Convolutional network
- Variational auto-encoder
- Generative adversarial network
- Automatic generation (mirror) of encoder/de-convolutional blocks.   

## Information Geometry
__Information geometry__ applies the principles and methods of differential geometry to problems in probability theory and statistics [ref 3]. It studies the manifold of probability distributions and provides a natural framework for understanding and analyzing statistical models.     
The directory 'informationgeometry' focuses on the __Fisher Information Metric__ (FIM).    
    
The Fisher Information Matrix plays a crucial role in various aspects of machine learning and statistics. Its primary significance lies in providing a measure of the amount of information that an observable random variable carries about an unknown parameter upon which the probability depends on.   
The Fisher information matrix is a type of __Riemannian metric__ that can be applied to a __smooth statistical manifold__. It serves to quantify the informational difference between measurements. The points on this manifold represent probability measures defined within a Euclidean probability space, such as the Normal distribution. Mathematically, it is represented by the Hessian of the __Kullback-Leibler__ divergence.

## Fractal Dimension
Configuring the parameters of a 2D convolutional neural network, such as kernel size and padding, can be challenging because it largely depends on the complexity of an image or its specific sections. __Fractals__ help quantify the complexity of important features and boundaries within an image and ultimately guide the data scientist in optimizing his/her model.    

A __fractal dimension__ is a measure used to describe the complexity of fractal patterns or sets by quantifying the ratio of change in detail relative to the change in scale.    
For ordinary geometric shapes, the fractal dimension theoretically matches the familiar Euclidean or __topological dimension__.    

There are many approaches to compute the fractal dimension of an image, among them:
- Variation method
- Structure function method
- Root-mean-square method
- R/S analysis method
- Box counting method

The directory 'fractal' contains the implementation of the __box counting method__ for images and 3D objects.

## Markov Chain Monte Carlo
Sampling sits at the core of data science and analysis. The directory __mcmc__ explores a category of numerical sampling techniques, known as Markov Chain Monte Carlo, and how they can be implemented via reusable design patterns, using the __Metropolis-Hastings__ model as an example.     

## Signal Processing
A Kalman filter serves as an ideal estimator, determining parameters from imprecise and indirect measurements. Its goal is to reduce the mean square error associated with the model's parameters. Being recursive in nature, this algorithm is suited for real-time signal processing.    
   
The directory __control__ contains the implementation of __Kalman__ filters.
    
![KalmanFilter](images/Kalman_Filter.png)

# Reusable Neural Components Design
## Neural Blocks
Block is defined as a logical grouping of neural components, implemented as Pytorch __Module__. All these components are assembled into a sequential set of torch modules.   
```
class NeuralBlock(nn.Module):
    def __init__(self, block_id: Optional[AnyStr], modules: Tuple[nn.Module]):
        super(NeuralBlock, self).__init__()
        self.modules = modules
        self.block_id = block_id
```
    
For instance, a Convolutional block may include a convolutional layer, kernel, batch normalization and possibly a drop-out components of type __Module__.
```
class ConvBlock(NeuralBlock):
    def __init__(self, _id: AnyStr, conv_block_builder: ConvBlockBuilder) -> None:
        self.id = _id
        self.conv_block_builder = conv_block_builder
        modules = self.conv_block_builder()
        super(ConvBlock, self).__init__(_id, tuple(modules))
```
Convolutional neural block are assembled through a __builder__.
```
class ConvBlockBuilder(object):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 input_size: int | Tuple[int, int],
                 kernel_size: int | Tuple[int, int],
                 stride: int | Tuple[int, int],
                 padding: int | Tuple[int, int],
                 batch_norm: bool,
                 max_pooling_kernel: int,
                 activation: nn.Module,
                 bias: bool):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.batch_norm = batch_norm
        self.max_pooling_kernel = max_pooling_kernel
        self.activation = activation
        self.bias = bias
        
     def __call__(self) -> Tuple[nn.Module]:
        modules = []
        # First define the 2D convolution
        conv_module = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=self.bias)
        modules.append(conv_module)

        # Add the batch normalization
        if self.batch_norm:
            modules.append(nn.BatchNorm2d(self.out_channels))
        # Activation to be added if needed
        if self.activation is not None:
            modules.append(self.activation)

        # Added max pooling module
        if self.max_pooling_kernel > 0:
            modules.append(nn.MaxPool2d(kernel_size=self.max_pooling_kernel, stride=1, padding=0))
        modules_list: List[nn.Module] = modules
        return tuple(modules_list)
```
A Neural block can be __inverted__. A convolutional block can be automatically converted into a de-convolutional block.      
![ReusableNeuralBlocks](images/Convolution_Mirror.png)
    
The current hierarchy of neural blocks is defined as:
![NeuralBlock](images/Neural_Block_Hierarchy.png)

## Neural Models
Neural models are dynamic sequence of neural blocks that are assembled and converted into a sequence of torch __Module__ instances.   
The Base class for Neural model is defined as     
```
class NeuralModel(torch.nn.Module, ABC):
    def __init__(self, model_id: AnyStr, model: torch.nn.Module):
        super(NeuralModel, self).__init__()
        self.model_id = model_id
        self.model = model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
```
Each model inherits from __NeuralModel__ (i.e. Convolutional neural network type : __ConvModel__)

```
class ConvModel(NeuralModel, ABC):
    def __init__(self,
                 model_id: AnyStr,
                 conv_blocks: List[ConvBlock],
                 ffnn_blocks: Optional[List[FFNNBlock]] = None):
        ConvModel.is_valid(conv_blocks, ffnn_blocks)
        
        self.conv_blocks = conv_blocks
        self.in_features = conv_blocks[0].conv_block_builder.in_channels
        self.out_features = ffnn_blocks[-1].out_features
 
        # 1. Assemble the convolutional modules
        modules: List[nn.Module] = [module for block in conv_blocks for module in block.modules]
        
        # 2. Assemble the fully connected modules if necessary
        if ffnn_blocks is not None:
            self.ffnn_blocks = ffnn_blocks
            modules.append(nn.Flatten())
            [modules.append(module) for block in ffnn_blocks for module in block.modules]
        super(ConvModel, self).__init__(model_id, nn.Sequential(*modules))
```
The current class hierarchy for Neural models is defined as:   

![NeuralModel](images/Neural_Model_Hierarchy.png)

## Environment

| Library         | Version |
|:----------------|:--------|
| Python          | 3.12.3  |
| SymPy           | 1.12    |
| Numpy           | 2.1.3   |
| Pydantic        | 2.4.1   |
| Shap            | 0.43.0  |
| torch           | 2.5.1   |
| torchVision     | 0.20.1  |
| torch-geometric | 2.6.1   |
| torch_sparse    | 0.6.18  |
| torch_scatter   | 2.12    |
| torch_cluster | 1.6.3   |
| Scikit-learn    | 1.5.2   |
| Geomstats       | 2.8.0   |
| Jax | 0.4.34  |
| PyTest | 8.3.3   |
| matplotlib | 3.9.2   |



# References
-[Deep learning with reusable neural blocks](http://patricknicolas.blogspot.com/2023/03/building-bert-with-reusable-neural.html)    
-[Geometric Learning in Python - Basic](https://patricknicolas.blogspot.com/2024/02/introduction-to-differential-geometry.html)    
-[Differentiable Manifolds](https://patricknicolas.blogspot.com/2024/03/geometric-learning-in-python-manifolds.html)    
-[Differential Operators in Python](https://patricknicolas.blogspot.com/2023/12/explore-differential-operators-in-python.html)    
-[Intrinsic Representation](https://patricknicolas.blogspot.com/2024/03/geometric-learning-in-python-coordinates.html)   
-[Vector and Covector Fields](https://patricknicolas.blogspot.com/2024/04/geometric-learning-in-python-vector.html)    
-[Vector Operators](https://patricknicolas.blogspot.com/2024/04/geometric-learning-in-python-vector_3.html)   
-[Non-linear Functional Data Analysis](https://patricknicolas.blogspot.com/2024/04/geometric-learning-in-python-functional.html)   
-[Riemann Metric and Connection](https://patricknicolas.blogspot.com/2024/04/geometric-learning-in-python-riemann.html)   
-[Riemann Curvature Tensor](https://patricknicolas.blogspot.com/2024/04/geometric-learning-in-python-riemann_18.html)   

