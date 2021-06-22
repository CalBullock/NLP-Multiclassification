Python script to read reviews from an input excel file, and return a new excel file with predictions of their requests.

Takes the input excel file and the trained model as inputs.
Converts the review text to BERT embeddings.
Predicts outcome from embeddings.
Predicted probabilities saved to separate columns.
Returns .xslx file
