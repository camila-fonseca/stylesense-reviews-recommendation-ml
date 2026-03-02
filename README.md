# Fashion Forward Forecasting

**Predicting product recommendations from customer reviews**

This project was developed as part of the Udacity Data Science
Nanodegree and extended to follow stronger modeling and validation
practices.

------------------------------------------------------------------------

## Problem

StyleSense, an online fashion retailer, has product reviews where some
customers did not fill in the "Would you recommend this product?" field.

Although the recommendation label is missing in some cases, the review
text and structured information (age, product category, etc.) are
available.

The objective is to predict whether a customer would recommend a
product, with particular attention to detecting dissatisfied customers.

------------------------------------------------------------------------

## Dataset

This project uses the **Women's E-Commerce Clothing Reviews** dataset.

**Source:**\
Kaggle -- Women's E-Commerce Clothing Reviews\
https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews

### Dataset Characteristics

-   \~23,000 customer reviews
-   Binary target: `Recommended IND` (0 = not recommended, 1 =
    recommended)
-   Structured features:
    -   `Age`
    -   `Department Name`
    -   `Division Name`
    -   `Class Name`
-   Text feature:
    -   `Review Text`
-   Imbalanced target distribution (majority class = recommended)

### How to Reproduce

1.  Download the dataset from Kaggle.
2.  Place the CSV file inside: data/raw/
3.  Run the notebooks in order.

------------------------------------------------------------------------

## Approach

This is a binary classification problem with class imbalance.

Instead of optimizing for overall accuracy, this project focuses on
performance for the minority class (customers who would **not**
recommend the product).

### Modeling Pipeline

-   Numerical, categorical and text features handled inside a single
    scikit-learn `Pipeline`
-   `ColumnTransformer` for heterogeneous preprocessing
-   TF-IDF vectorization for review text
-   Logistic Regression as baseline model
-   Cross-validation for hyperparameter tuning
-   Custom scoring using **F0.5 for class 0**
-   Decision threshold optimization based on validation performance

All preprocessing steps are included inside the pipeline to prevent data
leakage.

------------------------------------------------------------------------

## Evaluation Strategy

The modeling process follows three stages:

1.  Baseline model -- Logistic Regression with default threshold (0.5)\
2.  Hyperparameter tuning -- RandomizedSearchCV using F0.5 for class 0\
3.  Threshold optimization -- Validation-based selection of the optimal
    probability cutoff

The test set is used only once for final evaluation.

Additional diagnostics include:

-   Precision--Recall Curve (class 0 treated as positive)
-   Average Precision score
-   Comparison between baseline, tuned model, and tuned + threshold

------------------------------------------------------------------------

## Key Takeaways

-   Under class imbalance, hyperparameter tuning alone may not improve
    minority-class performance under a fixed threshold.
-   Decision threshold selection can significantly impact
    business-aligned metrics.
-   Separating hyperparameter tuning, threshold selection, and final
    testing helps avoid data leakage and optimistic bias.

------------------------------------------------------------------------

## Limitations

-   Logistic Regression assumes linear separability.
-   TF-IDF does not capture deeper semantic meaning.
-   Threshold was selected on a single validation split.
-   No probability calibration was applied.

Possible extensions:

-   Probability calibration\
-   More expressive models\
-   Cross-validated threshold selection

------------------------------------------------------------------------

## Tech Stack

-   Python\
-   pandas / numpy\
-   scikit-learn\
-   spaCy\
-   Matplotlib

------------------------------------------------------------------------

## Repository Structure

data/ notebooks/ 01_eda.ipynb 02_modeling.ipynb src/ data.py features.py
evaluate.py train.py

------------------------------------------------------------------------

## License

This project is licensed under the MIT License.\
See the LICENSE file for details.

------------------------------------------------------------------------

## Author

Camila Fonseca\
LinkedIn: https://www.linkedin.com/in/camila-fonseca/
