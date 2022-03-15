# Context

This project was completed as part of the General Assembly Data Science Immersive course. I present here a comprehensive overview of the machine learning pipeline.

# Table of Contents

- [Background](#1-background)<br>
- [Problem Statement](#2-problem-statement)<br>
- [Goals](#3-goals)<br>
- [Data](#4-data)<br>
  - [Data Acquisition](#41-data-acquisition)<br>
  - [Data Cleaning](#42-data-cleaning)<br>
  - [EDA](#43-eda)<br>
- [Approach to Modeling](#5-approach-to-modeling)<br>
- [Modeling](#6-modeling)<br>
  - [Results](#61-results)<br>
  - [Evaluation](#62-evaluation)<br> 
- [Limitations](#7-limitations)<br>
- [Conclusions](#8-conclusions)<br>
- [Further Work](#9-further-work)<br>
- [Libraries Used](#10-libraries-used)<br>
- [Contact](#11-contact)<br>

# 1. Background

Scientific research is done by individuals or teams of scientists and they write their research up and they submit it to be published in a peer-reviewed journal. This ensures that there is a chance to check no one’s method is flawed or results have been completely fabricated; more minds are generally better than one and as a scientist, your research really needs to stand up to scrutiny. This system is not perfect, there is bias all the way up the peer review ladder and career politics exists but another aspect that no scientist looks forward to is the time and tedium involved in the submission-revision-resubmission process, especially if you go around the loop a few times only to be ultimately rejected.


# 2. Problem Statement

What is the problem here? It is hard to get published as a scientist and the process is time-consuming and expensive. 

It can cost in the hundreds of pounds to have your article peer reviewed and then the more prestigious open access journals require payments in the thousands of dollars because they don’t make any money from subscriptions and the reward for research scientists is prestige. An exacerbating factor to this exists because of the way funding works: scientists need to write to grant bodies which provide the funding, and this results in quite heavy competition for funding.



# 3. Goals

My goals were to build a tool that could tell you which journal or journals you should submit your manuscripts to. This would hopefully save scientists time and money. I was hoping to achieve this by using a classification model to predict the most suitable journal for submission. Now is a good time to mention that I was naive to think that I could achieve this goal with my data, the story of this project is one of compromise. Problems will be discussed, however it is appropriate to mention that my main goal required shifting; I decided I wanted to see if it was possible to use my data to train a classification model on manuscript data to predict the scientific field that a manuscript belongs to.

Specifically, my goal is to be able to train a model that can predict an articles’ scientific domain accuracy higher than baseline.

I still wanted to build a tool that scientists could use to tailor their peer-review process, I just changed tack and said “if I can train this model to classify scientific articles into their respective scientific field, I’ll then try to make a set of models, each one aiming to classify more precisely which journal within a scientific field a manuscript belongs in”. 


# 4. Data

## 4.1. Data Acquisition

My data was sourced by web scraping https://pubmed.ncbi.nlm.nih.gov/. This site hosts a large database of scientific articles with a focus on medicine (and to a lesser extent, biology). I chose this site for a few reasons. Firstly, it was easy to scrape, I didn’t trigger any anti-scraping protocols and it was easily iterated through in a loop by simply incrementing an article number after the “.gov/” . Secondly, it had a lot of meta-information like tags and actually had the article abstract in plain text HTML instead of a picture, or slideshow which wasn’t possible to parse through.

Just to touch on the features of a scientific article, abstracts can be thought of as summaries of the contents of a scientific article. There can be many authors and these people will be affiliated with academic institutions. 



The problem that caused me to shift my main goal was that there were far too many journals to be used as classes to predict - there were 10771 unique journals and so I needed to rethink my approach.

In order to approach this as a classification problem I needed to feature engineer a target, which was achieved in two phases. Firstly, I web scraped a wikipedia list of journals that publish in a specific scientific field - there were 9 of these wiki lists, 4 of the more niche were absorbed into the larger fields and ultimate final classes (Medicine, Biology, Chemistry, Physics). I then looked up the journal's full name in a separate PubMed database and used this to match entries in the journal column to a scientific domain, as my data for the journal was limited to the abbreviation (e.g. Proc R Soc B = Proceedings of the Royal Academy B). This phase labeled about 50% of the raw data points and involved some tricky dictionary lookup.  Secondly, I manually went through and labeled the most commonly occurring unlabelled journals. This phase was very time-consuming, and I found that eventually a point of diminishing returns is reached in terms of value of time spent; I was at a point where essentially 19 of out 20 unlabelled journals were fringe medical journals. I, therefore, settled with labeling half the unique journals (5380) and this captured about 75% of the data points that survived general cleaning. I then dropped the unlabelled data points.

I still had severe class imbalance, I mentioned this database focussed on medicine and this really comes through in the data. We see medical articles comprise 55% of the dataset, with biology following at 25% and then there is a drop off.



I attempted to deal with my class imbalance in a few ways. I attempted to use SMOTE to upsample minority classes, however I found that this ballooned the size of my dataset and resulted in models that took infeasible amounts of time to run. I also was running into memory issues. I therefore decided to run models on a few versions of the dataset:

* Full data (untreated)
* Random under-sampling of majority classes
* Undersampling majority classes to frequency of minority class


These were ultimately all unfruitful in terms of achieving my specific goal(s), however the full data set appeared to generate the most successful models, thus this was the dataset I tried to optimize my models on.



## 4.2. Data Cleaning

Significant amounts of text processing using regex was required in cleaning the web scraped data. I have given one example below but the associated project notebook details every instance that regex is used as well as notation for whatever the regular expression captures.



Additional standard cleaning steps (e.g. backfilling missing values, dropping NAs etc.) were also undertaken and can be found in the project notebook file but do not warrant extended discussion.

## 4.3. EDA

It was during the exploratory data analysis phase that I uncovered many of the problems with my dataset (i.e. too many unique labels, class imbalance once custom labels applied) so in this section I will focus only on the plots, but I will detail later the aspects of my approach that were influenced by what I discovered about the data during EDA because, to me, the way in which certain features of the data influence one’s approach is slightly more interesting and important than simply reporting those features.



These wordclouds use the size of the word to show the relative frequency of the top 100 most commonly occurring words in the various classes and columns, hopefully it should be clear from these wordclouds that there is a difference in text when comparing the different scientific domains. I have also produced these wordclouds for the other text columns but for clarity I am only showing here the most common words in the abstract column.



# 5. Approach to modeling

The features I ended up using from my data were: Title, Abstract, Tags, Year of publication. I dummied the year which I extracted from the date column, as year was the only feature of date that was present in all data points. The other three predictors were text. This involved natural language processing or NLP, this involved vectorizing the text columns. I used TF-IDF vectorization as I wanted to account for the relative importance of words as well as the count (i.e. count vectorization, which I also tried implementing and performed worse than TF IDF vectorization).
I would have liked to use author data as well as article type as predictors. For authors,t I was planning to use this to find the h-index of authors, which accounts for how much a researcher publishes and how much others use that work; it’s essentially an index of how good you are compared to your peers. This seemed like a really helpful predictor but conceptualizing the transformation of this feature of the data into a single feature (or uniform set of features across all rows) was incredibly difficult; no justification seemed to be without significant and obvious bias (against papers with lots of authors, for papers with one mentor and many students etc.) and I would be very open to ideas on how to use this data as a predictor of journal (or journal field).
For article type it was a case that there weren’t enough articles that actually had put this information into the PubMed database.

Ultimately the target labels I settled on were Medicine, Biology, Physics and Chemistry; this represents quite a reduction from over 10 thousand! As mentioned earlier, I have had to change the class label of one class originally termed ‘general_journal’. These are rather a special case because they had their own wiki list, they do publish non-specific scientific advancements, but almost all focus the minutiae of their activities on life sciences, which can be thought of basically as biology. I originally modeled on all 5 target labels but my first models were really overfitting on this category as you can see from this classification report.


# 6. Modeling

## 6.1. Results

These are the models I ran on my data:

Neighbors Classifier (K Nearest Neighbors)
Linear model Classifier (Logistic Regression)
Tree Classifier (Decision Tree Classifier)
Bagging Classifier (Random Forest Classifier)
Gradient-boosted Classifier (Gradient Boosting Classifier)
To reiterate: my specific goal was to be able to train a model that can predict an articles’ scientific domain accuracy higher than baseline. The baseline accuracy for my data set was 0.550 (to 3.s.f).



Overall my results were extremely disappointing. I tried fitting these models on the different versions of my data - the full data, the randomly downsampled subset and the subset with every class downsampled to the minority class. I ultimately found the best results when I was training on the whole dataset; I think this is because this data set captures the most variance in text nuance.

The most time-consuming variable to test was the number of  features to include when performing TF-IDF vectorization; for all text columns I varied the number of features from between 0 (i.e. the column was not included in modeling) and 3500, which in this case simply means that each text column is dummied into 3500 columns, with one column representing one word.

I additionally tried 3 different train-test splits (50/50, 70/30, 80/20) and the best performing models ended up being those that were fit on the highest number of rows (i.e. the 80/20 split seemed to train the best performing models).

For my K Nearest Neighbors models, the confusion matrices that came out of this seemed to have what I would call arbitrary assignment, though there was always noticeable recall in the medicine category. These models typically performed worse than the other classification models.

For my logistic regressions - These were some of the more successful models (though that is not saying much). I specifically wanted to apply lasso regularization due to the large number of columns that NLP generates and initially chose quite a high regularization strength.  I will say that it is my understanding that while Logistic Regression is not classification, it can be used to model classification, as classes can be assigned based on the multivariate log-likelihoods that logistic regression generates.

Decision Trees seemed to be almost as good as Logistic Regressions. They suffered from the problem that a baseline score can be achieved by predicting everything as the majority class. This is obviously appealing if models are struggling to even achieve baseline accuracy but not appealing to me. These models seemed to be the best at recalling the lower proportioned classes - chemistry and physics, though this is just an observation and was never quantified in this project.

Concerning ensemble methods, for my bagging models, what is interesting is that these models seem the most inclined to learn that they can increase their accuracy simply by predicting everything as 'medicine'. It is also interesting that these models also seem intent on misclassifying enough rows that they don't ever technically outperform Logistic Regression. I wanted to investigate boosting because I was hopeful that this might help bump my model accuracy to higher than baseline, but this was similarly hopeless.

I then used gridsearch methods to setup a range of hyperparameters to iterate through in hopes of improving the accuracy of my logistic regression model; I recognise that this is somewhat of a redundant step given none of my models were able to achieve an accuracy higher than baseline, however it is important to undertake these steps of the ML pipeline as these are necessary steps to complete when a project is successful.




## 6.2. Evaluation

Ultimately, gridsearch methods to tune hyperparameters for my best model resulted in an accuracy of 0.548 (to 3.s.f.). This model was still unable to achieve an accuracy higher than baseline (0.550).

I’d like to highlight three components of this: 

#### Success Metrics

Below you can see the confusion matrix of my best mode. We can see that, essentially, the medicine class is just being predicted. We see some small correct predictions in other classes but it’s not close and my best model ended up predicting zero chemistry articles as chemistry.



#### Coefficients 

Below we can see the 10 most important features for two of the classes. We can see that year of publication has a quite pronounced effect in my best model - excluding this made models worse but the extent to which it uses the dummy variables associated with year, I suspect my best model is overfitting on this column.

For this logistic regression, there were four lines plotted, one for each class; we have about 1400 coefficients for this, because this model used data that had had TF-IDF performed on the text columns to select the 800 most important words in the tags column, the 400 most important words in the abstract column, the 100 most important words in the tags column. Really what’s interesting is the prominence of medical words in non-medical categories, sometimes this is because it’s saying ‘these words existing in the datapoint reduce chances of being a blank, say, chemistry, journal, but the reason I’ve shown the tables instead of a plot is just to highlight the numbers of the coefficients: there is no obvious trend, they’re very low and no coefficient is orders of magnitude larger than any others, which would at least have been remarkable.

#### Precision-recall and ROC curves

The precision-recall curves paint the same picture as the confusion matrix. The ROC curve below shows how classification changes as the threshold probability for prediction changes and then the precision recall curve shows how precision and recall change. In a multi-class classification setup, micro-average is preferable if you suspect there might be class imbalance. I did have a relatively high micro-average compared to the rest of the curves, so it’s possible that this model can be improved.




# 7. Limitations

Concerning the integrity of the data I think there is low risk of leakage and bias - my data has medicine as the majority class but in truth the proportions of my classes somewhat mirror the number of journals in each field and the absolute amount of literature that gets published, so medicine just publishes more and faster than other fields. Leakage concerns arise in the custom labeling phase (i.e. phase one of feature engineering a custom label) as I simply do not know exactly what was being labeled in the procedural lookup phase. The second phase (manual lookup via googling journal name) holds relatively low risk of leakage and I can say more certainly that each journal was labeled correctly as I was doing it personally and I know the slightly less obvious ones simply took more time to deduce the appropriate label. 

All data was sourced from the same place, no source of leakage or bias among the dataset from sourcing. I have tried different splits for train-test (50-50, 70-40, 80-20) and all seem unable to reach an accuracy of baseline, the most successful models are the ones that happen to learn to predict almost everything as the majority class and thus approaches baseline. 

# 8. Conclusions

Given that my models were unable to achieve an accuracy higher than baseline my conclusion is obvious: the data that I have cannot be used to answer the questions I have asked: Can we classify scientific articles into their respective domains? No.

Why is this? I think personally that there is more difference in corpus composition within scientific fields than there is between them at the level of granularity to which I examined. I think the problem with this question is: you need loads of features (but then computing time and memory is an issue) and you risk overfitting. Even though it looked promising from the wordclouds, the absolute abundance, relative to the absolute number of words in the corpus, is a drop in the ocean. Surely, then, you have to only look at a few words, right? Unfortunately the result of that is underfitting - the model has no idea what to do with most of the data because the text features you train on simply don't exist in most data points; in a sense you trade off quantity for quality and I’m not sure either end of the trade-off accomplishes anything meaningful.

# 9. Further Work

In future versions of this project, I would like to look at within-field journals and try to classify these. It would be more practical as a project if I was approached by an individual who could present a list of the journals they were thinking of sending their manuscript to, as then we could start with a dataset with only these journals as targets. This hopefully will demonstrate more nuance and difference in the corpus composition.

I would really like to investigate a multinomial logistic regression with this as I believe we can apply classifications based on the target probability and thus can also generate likelihoods for other targets (if say, the individual wanted to know which 5 submission processes out of 20 journals they should use the rest of their budget on).

I’d also like to do the rest of the things I’ve mentioned, so using predictors I couldn’t figure out how to use in the time I had as well as maybe making it a binary classification and have the target represent a group of the most reputable journals, which I imagine would be an attractive tool for my target audience (research scientists).

I’d also like to try to use different packages to accomplish the same goals and see if this influences the performance of my best model. I think I missed a trick in that I didn’t apply a log transformation to my data. This revelation came after watching a presentation on another text-based ML project and I noticed that frequency of words used in a corpus tends to follow an exponential distribution, with the most common words being used orders of magnitude more than the less commonly occurring words. I would also like to try to use n grams in my TF-IDF vectorization; I think this could be especially rewarding when considering the nature of keyword tags, often the significance comes from words being next to each other. I also found a library, gensim, which is used to process text for ML and I would like to investigate the effects of using this instead of the vectorization capabilities built-into sklearn.


# 10. Libraries used

<b>Web scraping</b>
  
  * requests
  * BeautifulSoup
  * Re
  * Time
  * Random
  
<b>Useful data wrangling</b>
  
  * Pandas
  * NumPy

<b>Modeling and preprocessing</b>

  * SKLearn
  * NLTK
  
<b>Data visualization</b>
  
  * Matplotlib
  * Scikitplot
  * Seaborn
  * WordCloud

# 11. Contact

If you found this project interesting or would like reach out, you can [find me on Linkedin](https://www.linkedin.com/in/ross-birkinshaw-102701215/).
