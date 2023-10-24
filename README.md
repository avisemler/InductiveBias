# InductiveBias
Investigating inductive bias of various training methods.

Planned training types:

- random features + linear classifcation: use an untrained neural network with a random intialisation to extract features from the data and then train a linear classifier on those features.
- SGD with typical cost function
- SGD optimising for orthogonality/usefullness of features

Plan for comparing inductive bias:
- probability that various generalisations are made (discretising by clasification for selected examples)
