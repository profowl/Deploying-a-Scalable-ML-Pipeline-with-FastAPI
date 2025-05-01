# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model is a binary classifier trained to predict whether an individual's income exceeds $50,000 per year based on demographic features such as education, occupation, race, and sex. It is built using a standard machine learning pipeline with data preprocessing, categorical encoding, and a supervised learning algorithm (e.g., logistic regression, random forest, or similar). The model was developed using the UCI Adult Census Income dataset.

## Intended Use
This model is intended for educational, research, and prototyping purposes, particularly in the domain of fairness and model evaluation. It can be used to explore issues such as bias, feature importance, and performance disparities across demographic slices. The model is not intended for deployment in real-world applications without additional validation and fairness audits.

## Training Data
The model was trained on the UCI Adult dataset, which consists of census data from the 1994 U.S. Census database. The dataset includes 32,561 examples in the training split. Each record includes features such as age, education, race, sex, occupation, workclass, and income label. Some categorical values are missing and represented with '?', which were handled as part of preprocessing.

## Evaluation Data
Evaluation was performed on a held-out validation set from the same distribution as the training data. Slice-based performance analysis was conducted to evaluate fairness and consistency across different demographic segments such as education levels, race, sex, occupation, and marital status.

## Metrics
The primary metrics used to evaluate model performance are Precision, Recall, and F1 Score. These metrics are reported both globally and across slices of individual features to assess fairness and robustness.

__*Summary of Performance on Selected Slices:*__
- **Sex:**
    - Male: F1 = 0.6997
    - Female: F1 = 0.6015
- **Race:**
    - White: F1 = 0.6850
    - Black: F1 = 0.6667
    - Asian-Pac-Islander: F1 = 0.7458
    - Amer-Indian-Eskimo: F1 = 0.5556
    - Other: F1 = 0.8000
- **Education:**
    - Bachelors: F1 = 0.7404
    - HS-grad: F1 = 0.5261
    - Masters: F1 = 0.8409
    - Some-college: F1 = 0.5914
    - 7th-8th: F1 = 0.0000
    - Doctorate: F1 = 0.8793
- **Occupation:**
    - Exec-managerial: F1 = 0.7736
    - Prof-specialty: F1 = 0.7778
    - Handlers-cleaners: F1 = 0.4211
    - Other-service: F1 = 0.3226
- **Workclass:**
    - Private: F1 = 0.6856
    - Self-emp-inc: F1 = 0.7672
    - Federal-gov: F1 = 0.7914
- **Marital Status:**
    - Married-civ-spouse: F1 = 0.7116
    - Never-married: F1 = 0.5641
    - Divorced: F1 = 0.4967

## Ethical Considerations
The dataset and model may reflect historical biases and structural inequalities present in U.S. census data from 1994. For example, performance disparities between sex and race categories may indicate algorithmic bias or imbalanced representation in the dataset. Additionally, the use of sensitive attributes like race and sex in predictive modeling must be approached cautiously and ethically, particularly in high-stakes decision-making contexts such as hiring or lending.

This model is not suited for real-world deployment without thorough audits for fairness, explainability, and compliance with relevant legal frameworks such as the Fair Credit Reporting Act (FCRA) or the Equal Credit Opportunity Act (ECOA).

## Caveats and Recommendations
- The model shows lower performance on some minority or less-represented groups (e.g., *7th-8th education, Amer-Indian-Eskimo, Other-service occupation*), which may result in unfair predictions for those subpopulations.
- Categories with very low counts (e.g., *Armed-Forces, Hungary, Preschool*) exhibit perfect scores due to their small size, which may be misleading.
- It is recommended to further balance the dataset or apply fairness-aware learning techniques to mitigate bias.
- Users should not rely on this model for high-stakes decisions and should perform additional evaluation tailored to their use case.