# Titanic Survival Prediction using Naive Bayes

This project implements a custom Naive Bayes classifier from scratch to predict survival on the Titanic dataset. The implementation uses a lookup table method for categorical data handling.

## Overview

The classifier analyzes passenger attributes to predict survival probability, achieving approximately 76% accuracy on the test set. The model processes both numerical and categorical features through careful feature engineering.

## Features Used

- Sex (male/female)
- Passenger Class (1st, 2nd, 3rd)
- Age (converted to categories: Child, Teenager, Adult, Unknown)
- Port of Embarkation (S, C, Q)
- Number of Siblings/Spouses (SibSp)
- Number of Parents/Children (ParCh)

## Data Preprocessing

1. Age values are categorized into groups:
   - Child: ≤ 12 years
   - Teenager: 13-19 years
   - Adult: > 19 years
   - Unknown: Missing values

2. Categorical encoding:
   - Sex: male (0), female (1)
   - Embarked: S (0), C (1), Q (2)
   - Age_Group: Adult (0), Unknown (1), Teenager (2), Child (3)

3. Feature selection:
   - Removed 'Fare' column
   - Focused on categorical and discretized features

## Implementation Details

### Key Components

1. **Train-Test Split Function**
   - Implements random sampling for dataset division
   - Configurable test size ratio

2. **Lookup Table Creation**
   - Handles categorical data
   - Implements Laplace smoothing for rare values
   - Calculates class probabilities

3. **Prediction Function**
   - Uses multiplicative probability combination
   - Handles missing feature values gracefully

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- random
- time

## Usage

```python
# Load and prepare data
df = pd.read_csv("Titanic-Dataset.csv")
train_df, test_df = train_test_split(df, test_size=0.2)

# Prepare data for Naive Bayes
train_df = prepare_data(train_df)
test_df = prepare_data(test_df, train_set=False)

# Create lookup table
lookup_table = create_table(train_df, label_column="Survived")

# Make predictions
predictions = test_df.apply(predict_example, axis=1, args=(lookup_table,))

# Check accuracy
accuracy = (predictions == test_df.Survived).mean()
```




