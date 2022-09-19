# pyMGBoost - python Meta Gadient Boosting
`pyMGBoost ` is nothing more than a common interface for four popular gradient-boosting (ensemble) frameworks: XGBoost, CatBoost, LightGBM and scikitlearn Gradient Boosting. [Meta](https://en.wikipedia.org/wiki/Meta) (from the Greek μετά, meta, meaning "after" or "beyond") is a prefix meaning "more comprehensive" or "transcending." Similarly, the name was here used to mean that there are more than one gradient boosting implementation available. `metaGBoost` is shortened to `MGB`.

## Why Gradient Boosting?
Gradient Boosting is a form og ensemble that has become the de-facto and almost ubiquitous solution for tabular data. Thus, I thought that having a small wrapper around it to get myself off the ground quickly was a good project. So here it is the reason of why this reposotiry was created.

## CatBoost
- What is it? CatBoost is a machine learning algorithm that uses gradient boosting on decision trees.
- Main page can be accessed [here](https://catboost.ai/en/docs/)

## XGBoost
- What is it? 
- Main page can be accessed [here]()

## LightGBM
- What is it? 
- Main page can be accessed [here]()

## scikit-learn Gradient Tree Boosting
- What is it? 
- Main page can be accessed [here](https://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting)

## What does metaGBoost do?
Given a dataset and some user-dependent parametres run the different frameworks and compare them. Essentially automate the following bullet point:
- Build a default model
- Build a user-defined model
- Tune the model
- Get the learning curves
- Get the model uncertainty
- Get the metrics
- Build an average/blended version of the models. Average is here intended as the arithmetic mean, whereas blended is here here a model where the user provides the weights to each model.

## Regression & classification
- Regression - currently under development
- Classification - supported in the future

## Code Structure
The source code is split in several modules. I am aware that this may not the best options but it offer the most flexible way to code and test different options. I am open to suggestion on how to do it better from a sofware stand point. 
- MGB/MetaGradientBoosting.py
- MGB/Modules.py
- XGBoostWrapper.py
