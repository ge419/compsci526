#  DisHinge 

Setup
```
pip install torch huggingface numpy matplotlib seaborn pandas datasets
pip install debugpy ipython
```

Get all $N$ ingredients and make an undirected graph. 
Each of the $N$ nodes should also have a weight. 
Train the edge weights so that ingredients A-B weight is higher if A and B are used together more often. 
These weights are stored in a $N \times N$ matrix, and represent the preferences of the average human. 
How does our graph recommend a dish? So given some matrix $A \in \mathbb{R}^{N \times N}$ and weights $w \in \mathbb{R}^N$. There's 2 steps. 

1. Choose the ingredients. You train RBM, a restricted Boltzmann machine. Take any function $f: \mathbb{R}^n \to \mathbb{R}$. Now, we want to define a probability distribution. How? 
$$
   p(x) \propto e^{-f(x)}
$$
We can encode the set of ingredients $S \subset \{1, \ldots, N\}$ as a vector $x \in \{0, 1\}^N \subset \mathbb{R}^N$. For example, 
$$
   x = \begin{pmatrix} 1 \\ 0 \\ 1 \\ 0 \end{pmatrix}
$$
represents selecting ingredient $1$ and $3$. Then, the function $f$ will have the form 
$$
   f(x) = -w^T x - x^T A x
$$

2. Choose the dish based off of ingredients. Use a classical graph algorithm or approximate algorithm to do this. 

Do EDA. 

====

Then add a constraint where you give the app what ingredients you already have. So constrained optimization is trivial. 

You start accepting or rejecting the dishes it recommends, and based on your patterns, the model adjusts the weights. 
Have it record daily nutritional intake, along with macros. 
