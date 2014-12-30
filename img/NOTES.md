elo-learning-curve.png
----------------------
Learning curve of ELO model.

* X-axis -- number of training examples (up to 30 000, 20 000 examples were used as a cross-validation set).
* Y-axis -- RMSE

response-time.png
-----------------
Correlation between response time and the probability on recall. Each value in the boxes represents the probability that the user recalles a particular country.

pfa-spacing-first-answers-wrong.png
-----------------------------------
Grid Search for fitting the spacing rate and decay rate in PFA model extended with forgetting and spacing effects.

* Only the items which were answered incorrectly by the user in the very initial presentation have been used in the training of the model.
* Train set size: 3261
* Test set size: 1107
* AUC of the original model on this set: 0.78490910201
* RMSE of the original model on this set: 0.455674388602
