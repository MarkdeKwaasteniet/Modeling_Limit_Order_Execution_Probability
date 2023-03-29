# Limit Order Execution Probability Modeling in the Crypto-Asset Market

This repository contains the data analysis done for my thesis at BlockTraders, an Algorithmic Cryptocurrency Trader in The Hague.

## Conclusion of my research:
The paper studies limit order execution probability in the crypto-asset market by comparing
parametric and non-parametric models using high-frequency trading data. The models are
assessed on prediction accuracy, estimation time, and variable importance. The results indicate
that non-parametric models are more accurate in modeling the execution probability of limit
orders. The random forest and the kernel density estimation are the recommended models
when weighing both accuracy and estimation time, with order book quantity being the most
influential market factor for model accuracy.

## Sidenote regarding the Python Code

Most of the analysis is done in a Jupyter Notebook file whilst calling function from an additional Python file.
The main reason being that this analysis is only to be done once properly and is not meant as a continuous project. 
This explains the lack of structure in this repository and the large amount of code in one file as this proved to be most efficient when having a lot of iterations done during the analysis.


