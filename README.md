# NLP of game reviews

## Overview
Build an algorithm for classification of user reviews in to one of the four categories. The quality of the algorithm should be evaluated using hold-out subset or cross-validation technique.

## Functionality
 * Reads review strings from the csv input.
 * Reviews get unnecessary punctuction removed, and reviews with character length <4 are removed from the dataset.
 * Reviews are then vectorised using TF-IDF and BERT embeddings for the final model.
 * Created a script to apply the predicted outcome to each review in the xlsx format.
