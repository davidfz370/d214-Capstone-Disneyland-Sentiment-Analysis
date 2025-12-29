Research Question
Known as the “Happiest Place on Earth,” Disneyland California has earned #1 on U.S. News of the best things to do in Anaheim, CA. Disneyland receives visitors from all over the world, giving them an experience unlike any other. Whether it is for Star Wars Land or trying the variety of food options, there is something everyone can enjoy. As prices rise for tickets and other amenities such as parking, visitors are frustrated and feel that the price does not equate to the experience they receive from a day at Disneyland. Some visitors may view wait times as a waste of the day at the park and see the FASTPASS service as another high expense to enjoy the experience, especially if visitors come with their families and friends.  
(USNews)
Given these experiences, the central question remains: Despite all of this, are there ways for Disneyland to improve the experience of parkgoers? Are customers upset with long lines? Rising prices? Or customer service? One way to answer this is to hear from the parkgoers themselves. Viewing reviews is a great way to see the overall trend for negative and positive experiences. However, given thousands of reviews, it would be very time-consuming to go through each one, analyze trends individually, and separate the reviews into numeric categories. The business needs to understand where to improve for higher customer satisfaction and revisiting.
Research Question: How accurately can a neural network predict Disneyland California ratings (1-5) based on customer sentiment?
Null Hypothesis: The null hypothesis for this analysis is that the chosen neural network does not predict ratings with 80% or more accuracy based on the reviews.
Alternate Hypothesis: The alternate hypothesis is that the neural network chosen is highly accurate at predicting customer 1-5 ratings with an 80% or higher accuracy score.

Data Collection
The data collection involved researching different ways to extract data and finding an open-source dataset. The dataset used for this analysis comes from Kaggle.com and is titled "Disneyland Reviews." It comprises 42,000 reviews from TripAdvisor's California, Hong Kong, and Paris locations. In this study, we will only focus on the California location.
The features for this dataset include:
Review_ID: unique ID for the reviews made.
Rating: A range from 1 to 5, with one being the most unsatisfied and five being the most satisfied.
Reviewer_Location: the origin location of the reviewer.
Review_Text: The review made by the park visitor.
Disneyland_Branch: The Disneyland location (Paris, Hong Kong, California)
An advantage of using this dataset is that the user ‘Arush Chillar’ completed the web scrapping process and uploaded a CSV file to Kaggle. Another benefit is the large dataset, which will help train the model to make predictions and improve accuracy. A disadvantage of this dataset is that reviews come from TripAdvisor only. There may be some insight lost by not including Yelp and Google reviews. There were no challenges in finding a dataset for this study other than taking time to compare sources.

(Chillar) 

Data Extraction and Preparation
The data extraction and preparation were completed using Python and the Python libraries. 
Tools For Data Cleaning and Extraction: Numpy, Pandas, re, matplotlib, nltk, tensorflow.
Use cases
Pandas
Pandas will be used based on the capabilities that we will use on the imported data frame. We will use pandas to find errors in the review data frame. Pandas also give us the ability to remove and select rows and columns.
Pros:
It can handle larger datasets and has methods used for data cleaning, such as nulls and duplicates.
Cons:
It is dependent on other libraries and memory intensive.
(AltexSoft Editorial Team)
(Python)
Numpy
Numpy allows us to perform numerical calculations and works hand in hand with the Pandas library to perform data frame manipulation.
Pros:
It helps perform mathematical functions on the data frame and multidimensional array support.
Cons:
It can use large amounts of memory.
(AltexSoft Editorial Team)
Re
Re or regular expression will be used to prepare the data, allowing us to clean the review column and only use alphanumeric characters.
Pros:
Suitable for data cleaning and can search, match, and manipulate our review text.
Pattern matching.
Cons:
Re can lead to errors because of regex patterns.
Matplotlib
Matplotlib will be used to create graphs and later when we want to display predictions and accuracies.
Pros:
Provides the ability to create visuals for EDA and predictions.
Cons:
Compared to applications like Tableau and Excel, it is not as visually appealing.
(Karl)
Nltk
Known as the natural language toolkit, we will use NLTK to perform tokenization, lemmatization, and padding.
It has word_tokenize, which we will use to split the reviews into words.
WordNetLemmatizer will convert words to the root form. Ex: ‘running’ would be converted to ‘run’
Nltk will also be used to download Punkt.
Pros:
It gives the ability to use tokenization, lemmatization, and other text-processing functions.
Cons:
It can be slow for large datasets and consumes a lot of memory.
(LinkedIn)
(Python)
Tensorflow
Keras
Used to import Tokenizer, which will give unique values to words. For example, ‘wait’ and ‘fun’ will be assigned to 0 and 1. 
Pad_sequences will ensure the same input size, which prevents data loss and makes for efficient batching.
Pros:
It gives the ability to use machine learning models and is scalable.
Cons:
Complex framework.
(Geeksforgeeks)
(Keras Team)
SpaCy
Used to load models and have access to more methods for text processing and lemmatization.
Pros:
It is good to use when speed is needed.
Used with TensorFlow.
Cons:
It can consume much memory. 
(Python)
(LinkedIn)
