# quora-question-pair-similarity-detection

<h1> 1. Business Problem </h1>

<h2> 1.1 Description </h2>
<p>Quora is a place to gain and share knowledge—about anything. It’s a platform to ask questions and connect with people who contribute unique insights and quality answers. This empowers people to learn from each other and to better understand the world.</p>
<p>
Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term.
</p>
<br>
> Credits: Kaggle 

**Problem Statement**
- Identify which questions asked on Quora are duplicates of questions that have already been asked. 
- This could be useful to instantly provide answers to questions that have already been answered. 
- We are tasked with predicting whether a pair of questions are duplicates or not. 

<h2> 1.2 Sources/Useful Links</h2>

- Source : https://www.kaggle.com/c/quora-question-pairs
<br><br>____ Useful Links ____
- Discussions : https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb/comments
- Kaggle Winning Solution and other approaches: https://www.dropbox.com/sh/93968nfnrzh8bp5/AACZdtsApc1QSTQc7X0H3QZ5a?dl=0
- Blog 1 : https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning
- Blog 2 : https://towardsdatascience.com/identifying-duplicate-questions-on-quora-top-12-on-kaggle-4c1cf93f1c30

<h2>1.3 Real world/Business Objectives and Constraints </h2>

1. The cost of a mis-classification can be very high.
2. You would want a probability of a pair of questions to be duplicates so that you can choose any threshold of choice.
3. No strict latency concerns.
4. Interpretability is partially important.


<h2> 1.4 Mapping the real world problem to an ML problem </h2>

<h3> 1.4.1 Type of Machine Leaning Problem </h3>
<p> It is a binary classification problem, for a given pair of questions we need to predict if they are duplicate or not. </p>

<h3> 1.4.2 Performance Metric </h3>

Source: https://www.kaggle.com/c/quora-question-pairs#evaluation

Metric(s): 
* log-loss : https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
* Binary Confusion Matrix

<h1> 2. Approaching Problem </h1>

<h2> 2.1 Exploratory Data Analysis </h2>

<h3> 2.1.1 Basic overview of question pair similarity</h3>

- High level overview reveals that 63.08% question pair are not similar and 36.92% question pair are similar.
- There are total 537933 number of unique questions out of which 111780 questions appear more than once.
- Histogram with log-scale is used to visualise the question appearence counts. Maximum number of times a single question is repeated: 157.
- Null values are filled with blanks.

<h3> 2.1.2 Basic feature engineering </h3>

Before the data cleaning , some basic feature enginnering is done. Following features are constructed:

- Frequency of question id 1
- Frequency of question id 2 
- Length of question 1
- Length of question 2
- Number of words in Question 1
- Number of words in Question 2
- (Number of common unique words in Question 1 and Question 2)
- (Total num of words in Question 1 + Total num of words in Question 2)
- (word_common)/(word_Total)
- sum total of frequency of qid1 and qid2 
- absolute difference of frequency of qid1 and qid2 

Refer to  section 3.3 of **1. Quora EDA** notebook for more details.

<h3> 2.1.3 Preprocessing of text </h3>

- Preprocessing:
    - Removing html tags 
    - Removing Punctuations
    - Performing stemming
    - Removing Stopwords
    - Expanding contractions etc.
 
<h3> 2.1.4 Advance feature extraction (NLP and fuzzy features) </h3>

Definition:
- __Token__: You get a token by splitting sentence a space
- __Stop_Word__ : stop words as per NLTK.
- __Word__ : A token that is not a stop_word


Features:
- __cwc_min__ :  Ratio of common_word_count to min lenghth of word count of Q1 and Q2 <br>cwc_min = common_word_count / (min(len(q1_words), len(q2_words))
<br>
<br>

- __cwc_max__ :  Ratio of common_word_count to max lenghth of word count of Q1 and Q2 <br>cwc_max = common_word_count / (max(len(q1_words), len(q2_words))
<br>
<br>

- __csc_min__ :  Ratio of common_stop_count to min lenghth of stop count of Q1 and Q2 <br> csc_min = common_stop_count / (min(len(q1_stops), len(q2_stops))
<br>
<br>

- __csc_max__ :  Ratio of common_stop_count to max lenghth of stop count of Q1 and Q2<br>csc_max = common_stop_count / (max(len(q1_stops), len(q2_stops))
<br>
<br>

- __ctc_min__ :  Ratio of common_token_count to min lenghth of token count of Q1 and Q2<br>ctc_min = common_token_count / (min(len(q1_tokens), len(q2_tokens))
<br>
<br>

- __ctc_max__ :  Ratio of common_token_count to max lenghth of token count of Q1 and Q2<br>ctc_max = common_token_count / (max(len(q1_tokens), len(q2_tokens))
<br>
<br>
        
- __last_word_eq__ :  Check if First word of both questions is equal or not<br>last_word_eq = int(q1_tokens[-1] == q2_tokens[-1])
<br>
<br>

- __first_word_eq__ :  Check if First word of both questions is equal or not<br>first_word_eq = int(q1_tokens[0] == q2_tokens[0])
<br>
<br>
        
- __abs_len_diff__ :  Abs. length difference<br>abs_len_diff = abs(len(q1_tokens) - len(q2_tokens))
<br>
<br>

- __mean_len__ :  Average Token Length of both Questions<br>mean_len = (len(q1_tokens) + len(q2_tokens))/2
<br>
<br>


- __fuzz_ratio__ :  https://github.com/seatgeek/fuzzywuzzy#usage
http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
<br>
<br>

- __fuzz_partial_ratio__ :  https://github.com/seatgeek/fuzzywuzzy#usage
http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
<br>
<br>


- __token_sort_ratio__ : https://github.com/seatgeek/fuzzywuzzy#usage
http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
<br>
<br>


- __token_set_ratio__ : https://github.com/seatgeek/fuzzywuzzy#usage
http://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
<br>
<br>
 
- __longest_substr_ratio__ :  Ratio of length longest common substring to min lenghth of token count of Q1 and Q2<br>longest_substr_ratio = len(longest common substring) / (min(len(q1_tokens), len(q2_tokens))


<h3> 2.1.5 Analysis of extracted features </h3>

- Plotting word clouds
  - Creating Word Cloud of Duplicates and Non-Duplicates Question pairs
  - We can observe the most frequent occuring words
  
- Pair plot of extracted features
- Observed the distribution of features
- TSNE visualisation: It can be clearly observed that the two classes can be seperated nicely, with some overlapping. This suggests that the advance features are useful in some ways.

Visit Section 3.5 of **2. Quora_Prerpocessing** notebook for further details.

<h3> 2.1.6 Feature Importance </h3>

Used Random Forest to get an idea of important features among  all advance and basic features.

<h3> 2.1.7 Text Vectorization </h3>

- TFIDF Weighted Word2Vec and Simple TFIDF vectorizations are used.

<h3> 2.1.8 Machine learning models </h3>

- TFIDF weighted Word2Vec vectorization combined with advanced and basic features
  - Used a baseline random model- To get a picture about worst case scenarion. 
  - Logistic Regression with SGDClassifier with **log** loss
  - XGBoost with hyperparameter tuning using Optuna: Optuna used Tree of Parzen Estimators (TPE) to optimize hyperparameters. For more details , please go through any of the following:
  
    - https://www.automl.org/wp-content/uploads/2019/05/AutoML_Book_Chapter1.pdf
    - https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf
    
- TFIDF vectorization with advanced and basic features
  - XGBoost model


  <h3> 2.1.9 Model Performance </h3>
  
  
| Model  | Vectorizer | Log Loss | Class 0 Precision  | Class 1 Precision  | Class 0 Recall | Class 1 Recall  | 
| --- | --- | --- | --- | --- | --- | --- |
| Random Model | Weighted TFIDF W2V | 0.8869  | 0.632  | 0.37  | 0.509  | 0.499  |
| SGDClassifier(loss='log') | Weighted TFIDF W2V | 0.4366  | 0.771  | 0.37  | 0.897  | 0.591 |
| XGBoost (hyp tuned with Optuna) | Weighted TFIDF W2V | 0.3222  | 0.863  | 0.8  | 0.889  | 0.76  |
| XGBoost | TFIDF | 0.2958  | 0.878  | 0.823  | 0.901  | 0.786 |


