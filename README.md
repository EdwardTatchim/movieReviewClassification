# movieReviewClassification

This was a text classification competition - sentiment classification. Every record, in the train.csv file and later the kaggle dataframe, is a movie review extracted from IMDB. My goal was to classify the sentiment of each review into "positive" or "negative" using machine learning models and Python programming language.

Please see internal documentation of kaggle.py file for step-by-step explanation of code details.

Here is a general overview of the task at hand and dataset source:

  The training data (train.csv) contained 10,000 reviews, already labeled with 1 (positive sentiment) or 0 (negative sentiment). The test data (test.csv) contained 5,000 sentences   that were unlabeled. My prediction submissions to Kaggle's leader board were .csv (comma separated free text) files (sampleSubmission.csv) with header line "Id,Category"           followed by exactly 5,000 lines. In each line, there were exactly two integers, separated by a comma. The first integer was the line ID of a test sentence (0-5000), and the       second   integer was the category my classifier predicted; one of (0,1).

  I was allowed to make 10 submissions per day. Once I submitted your results, I got an accuracy score computed based on 50% of the test data. This score positioned me somewhere     on the leaderboard. The evaluation metric was the accuracy - so the higher the better.


Thanks to Maas et al. (2011) for providing the dataset.
