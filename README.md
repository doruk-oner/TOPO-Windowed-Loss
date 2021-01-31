# TOPO-Loss

This repository is the PyTorch implemeantation for the paper [Promoting connectivity of network-like structures by enforcing region separation](http://arxiv.org/abs/2009.07011). Doruk Oner, Mateusz Kozin ́ski, Leonardo Citraro, Nathan C. Dadap, Alexandra G. Konings, Pascal Fua.

![TOPO Loss computation](https://github.com/doruk-oner/TOPO-Windowed-Loss/blob/main/Images/TOPO_Loss.png)
Figure 1. **Computing TOPO Loss.** We first tile the ground truth annotation and the distance map computed by our network (1). We use the ground-truth roads to segment each tile into separate regions (2). When there are unwarranted gaps in the distance map, there is a least one path connecting disjoint regions such that the the minimum distance map value along that path is not particularly small. We therefore take the cost of the path to be that minimum value (3) and we add to our loss function a term that is the maximum such value for all paths connecting points in the two distinct regions (4). This penalizes paths such as the one shown here and therefore promotes the road graph connectivity.

For computation of the loss, [MALIS](https://github.com/TuragaLab/malis) library is used.

## Datasets
1. [RoadTracer Dataset](https://github.com/mitroadmaps/roadtracer/)
2. [DeepGlobe Road Dataset](https://competitions.codalab.org/competitions/18467)
3. [Massachusetts Roads Dataset](https://www.cs.toronto.edu/~vmnih/data/)
4. [Water Drainage Canals Dataset](https://search.proquest.com/docview/2478659343?pq-origsite=gscholar&fromopenview=true)
