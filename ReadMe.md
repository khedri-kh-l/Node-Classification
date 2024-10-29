Node Classification
--------------------------

Node classification is a machine learning task in graph-based data analysis, where the goal is to assign labels to nodes in a graph based on the properties of nodes and the relationships between them. In this report, we want to compaire effiecincy  some models such as **Graphconv(Pytorch-Geometric), Graphconv(DGL), SplineConv, and SSP**. Frist of all, we bring out summary of these architectures. Also, we mention the difference between GraphConv from DGL and PyTorch Geometric. 
We implement all the above architectures on **Cora** dataset. 

## Graphconv

GraphConv is a type of convolutional layer for graph structured data, implemented in the PyTorch Geometric library. It is based on the paper sc-https://arxiv.org/abs/1609.02907 "[Semi-Supervised Classification with Graph Convolutional Networks" by Thomas Kipf and Max Welling.


GraphConv uses a simple and efficient convolution kernel that is defined by the normalized adjacency matrix of the graph. GraphConv leverages the spectral properties of the graph Laplacian to filter the node features in the Fourier domain.

GraphConv does not have a separate encoder and decoder, as it is a single-layer operation that can be stacked to form a multi-layer graph neural network. However, one can think of the convolution kernel as a linear encoder that maps the node features to a higher-dimensional space, and the activation function as a non-linear decoder that maps the output features to a lower-dimensional space.



##  Difference between GraphConv from DGL and PyTorch Geometric

- The main difference between GraphConv from DGL and PyTorch Geometric is the way they handle the graph structure and the message passing. DGL uses a graph object to store the node and edge features, and a message passing interface to define the convolution operation. PyTorch Geometric uses tensors to store the node features and the edge indices, and a message passing class to inherit the convolution operation


- Another difference between GraphConv from DGL and PyTorch Geometric is the performance and scalability. DGL claims to have better performance and scalability than PyTorch Geometric, especially for large and heterogeneous graphs. PyTorch Geometric claims to have better performance and scalability than DGL, especially for small and homogeneous graphs.


## SplineConv

SplineConv is a type of convolutional layer for graph structured data, implemented in the PyTorch Geometric library. It is based on the paper "[SplineCNN: Fast Geometric Deep Learning with Continuous B-Spline Kernels, by Matthias Fey et al.]
SplineConv uses a continuous and differentiable convolution kernel that is defined by B-spline basis functions. B-splines are piecewise polynomial functions that can approximate any smooth curve. SplineConv leverages the properties of B-splines to efficiently filter geometric input of arbitrary dimensionality.

SplineConv consists of two main components: an encoder and a decoder. 

The encoder takes the node features and the edge pseudo-coordinates as inputs, and maps by a linear layer to the attention weights that define the latent graph.The encoder can be a graph neural network, such as GCN or GAT.

The decoder takes the attention weights and the edge pseudo-coordinates as inputs, and outputs the new node features that are obtained by applying the B-spline kernel along the edges.The decoder is a linear layer that maps the edge pseudo-coordinates to a lower-dimensional space, followed by a B-spline basis function that computes the kernel values. The new node features are computed by multiplying the kernel values with the attention weights, and aggregating the results by a mean or max operation



Applies the spline-based convolution operator
<p align="center">
  <img width="30%" src="https://user-images.githubusercontent.com/6945922/38684093-36d9c52e-3e6f-11e8-9021-db054223c6b9.png" />
</p>
The kernel function is defined over the weighted B-spline tensor product basis, as shown below for different B-spline degrees.


| <img src="https://user-images.githubusercontent.com/6945922/38685443-3a2a0c68-3e72-11e8-8e13-9ce9ad8fe43e.png" width="400"> | <img src="https://user-images.githubusercontent.com/6945922/38685459-42b2bcae-3e72-11e8-88cc-4b61e41dbd93.png" width="400"> |
|:---:|:---:|


 
 
## SSP-Master 

The model consists of three classes: Net_orig, CRD, and CLS. Net_orig is the original model, while CRD and CLS are modified versions of it. Net is the final model that combines CRD and CLS. All the classes use the GCNConv layer from PyTorch Geometric library.


•  Net_orig has two GCNConv layers. The forward method of Net_orig takes the node features and the edge indices as inputs, and applies the GCNConv layers with ReLU activation, dropout, and log-softmax functions. 

•  CRD is a modified version of Net_orig that uses only one GCNConv layer for the input features. The forward method of CRD takes the node features, the edge indices, and an optional mask as inputs, and applies the GCNConv layer with ReLU activation and dropout. 

•  CLS is another modified version of Net_orig that uses only one GCNConv layer for the output features. The forward method of CLS takes the node features, the edge indices, and an optional mask as inputs, and applies the GCNConv layer with log-softmax function. The output is a tensor of shape [num_nodes, num_classes] that contains the log-probabilities of each node belonging to each class. The mask argument can be used to select a subset of nodes for training or inference.

•  Net is the final model that combines CRD and CLS. The forward method of Net takes the node features, the edge indices, and the train mask as inputs, and passes them to the CRD and CLS modules. The output is a tensor of shape [num_nodes, num_classes] that contains the log-probabilities of each node belonging to each class. The train mask argument can be used to select the nodes that are used for training.



## Evaluation
|Dataset|Model       |Epoch| Time-inference| Model Size |Test Accuracy|
|--     |--          |--   |---            |--                      |         --  |
|Cora   |Graphconv(Pytorch-Geometric)  |100  |0.009 ± 0.007  |755424  |0.716± 0.017|
|Cora   |Graphconv(DGL)                |100  |0.010 ± 0.002  |746720                      |0.727 ± 0.008|
|Cora   |SplineConv                    |50  |0.004 ± 0.000 |2212576                       |0.910 ± 0.008|
|Cora   |SSP-Master                    |200    |0.0371  |738016                       |0.808 ± 0.77|



### Requirements

    python==3.10.6
    dgl==1.1.2
    pytorch==2.0.1+cpu
    torch-scatter==2.1.1+pt20cpu
    torch-sparse==0.6.17
    torch-geometric==2.4.0
