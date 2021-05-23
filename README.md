# DeepLearningProject

This is the project for the course of Deep Learning (A.Y. 2019/2020). 
The project consists in implementing geometric deep learning techniques to predict virality of hashtags in tweets.

## Project

What is **virality**? Virality, in its original meaning, refers to viruses that can only survive by continously spreading from one host to another in a parasitic manner.
Actually, many real life phenomena exhibit a spreading behaviour to which we can extend the notion of virality.

The ability to **predict the spreading potential of a signal** has evident benefits, for example providing a mean to prevent the spread of undesired phenomena such as diseases or fake news, but also allowing companies to exploit this information to improve their advertising campaigns.

**Graphs** serve as an useful abstraction to model real world situations, and are well suited to represent spreading patterns, but are a domain in which traditional deep learning approaches struggle. Therefore, we turned to techniques that try to generalize the machine learning models to non-euclidean domains: in the case of deep learning models, this is usually called **geometric deep learning**. In particular, in our project we leveraged and compared *Graph Convolutional Networks* and *Graph Attention Networks* on both synthetic and real data. You can find more details in the project [presentation](presentation.pdf).

The project implementation has been done on Colab, using PyTorch. All the code used is accessible on a copy of the [Colab notebook](Project.ipynb).

## Documentation

Check the presentation we gave of our project [here](presentation.pdf).

## Resources used

- [Google Colab](https://colab.research.google.com/)
- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric)

## Authors

- [Andrea Caciolai](https://github.com/caciolai)
- [Donato Crisostomi](https://github.com/crisostomi)
