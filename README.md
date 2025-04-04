
# Mulimodal Rating Analysis using NLP and Machine Learning

## Problem

Given a dataset of market-related features like returns, value indices, market indices and other unstructured features like text-transcripts, the aim is to create a Data Science solution that better explains the current ratings of the debt instruments. The features describe market as well as the company performance and thus, can go a long way in estimating the risk involved in investing in these bonds.

## Approach

The main approach to this problem was to create a ML based solution built on structured data as well as the text data provided. Through use of embeddings, sentiment can be derived from text using pre-built transformers or libraries and that is what would be fed to the main model as a feature. Additionally, topic modelling was explored to create more features from text.
For numerical features, relationships with the Ratings are explored and highly correlated features are to be used in the model. Along, the way, various techniques like PCA, scaling or log-transformation will be tested and the final optimum set of features will be passed.
In terms of model creation, 3 candidates will be created : a structured data only model, a text data only model and a mixed model. Multiple traditional ML models will be considered and evaluated through performance metrics, over-fitting, under-fitting and feature weights. And finally, the performance of the predicted ratings will be assesed with respect to the independent features.

## Setup Instructions

Follow the steps below to set up your environment.

### 1. Clone the repository:

```bash
git clone https://github.com/Mitul2991/crisil-credit-ratings-modelling.git
```

### 2. Set up the environment:

If you are using `conda`, create a new environment:

```bash
conda env create -f environment.yml
```

Alternatively, if you are using `pip`, install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Running the Code

Once the environment is set up, you can run the cells in the notebook to replicate the results.

## Key Findings and Highlights

- In terms of the target, we can see that ustret30ind, de_ratio, usind2CRSPMIV1, SPIndsprtrn and usind2CRSPMEV1 are positively correlated.
- This aligns with business logic, as for example, an increase in debt-to-equity ratio might be associated with poorer ratings.
- Also, monthusdcnt, ustreb1ind, pe_exi, debt_at and monthvwretd are moderately correlated to the encoded target in the negative direction.
- Similarly, something like a price to earnings ratio is decently inversly correlated to the ratings which means that at better ratings these ratios are higher and thus, we can infer that when the market starts doing better there is lower risk involved.
- The transcripts on manual inspection seem to be mostly positive as well and so anomaly detection was done to find outliers that might be a little different from the actual data.
- The text model classifies the statements into two groups and on inspecting the two groups, there seems to be a little difference between them. While one group seems to be confirm positive events happening, the other group seems to potray encouraging emotions and can be labelled as progressive but not confirmed as positive.
- The best performance of the model in terms of minimal over-fitting and good precision, recall and AUC, was observed when using the highly correlated features without any pre-processing. Using the SMOTE technique to oversample the minority classes helps in reducing the overfitting further.
- Candidate 1 best model (Log reg) : train auc : 0.91, test auc : 0.79, avg accuracy gap through CV (n=5) : 0.35
- Candidate 2 best model (RF) : train auc : 0.62, test auc : 0.58, avg accuracy gap through CV (n=5) : 0.05
- Candidate 3 best model (Log reg) : train auc : 0.92, test auc : 0.81, avg accuracy gap through CV (n=5) : 0.34
- On comparing these 3 candidates, the mixed model has the best performance metrics although it has high over-fitting. That can be attributed to a very small training set.
- Returns seem to be important when predicting ratings which might be reflective of market conditions. This thus can prove an indirect correlation with risk.
- Debt to EBIDTA ratio is a good financial health indicator and thus, it rightfully impacts the risk ratings.
- From the wordclouds created for each rating, neutral keywords are more common at lower ratings as compared to higher ratings and the inverse applies for positive keywords.
- Returns, market-indices that reflect the market and financial ratios like debt-invested capital ratio, price to earnings ratio that reflect the financial health influence the ratings and with more data a more robust model without any overfitting can be created that will better explain the actual ratings.
- In terms of the transcripts, more variety can be explored with more data and perhaps encountering more negative keywords might help explain the ratings better.
