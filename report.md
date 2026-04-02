# Machine Learning Project Report: Evaluating Supervised Learning Algorithms for Customer Churn Prediction

## 1. Selection of an Appropriate Dataset
The dataset selected for this project is the Telco Customer Churn dataset. This dataset was selected because it represents a highly complex, real-world business problem faced by subscription-based services. Customer churn directly impacts company revenue, making it a critical scenario to evaluate using supervised learning. It contains a diverse mix of numerical, categorical, and binary data, moving beyond simple tutorial datasets, and is publicly hosted and widely recognized.

## 2. Description of the Dataset
Source Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset comprises 7,043 instances (customers) and 21 attributes. The context involves understanding customer behavior to prevent them from leaving. Attributes include:
- Demographic Data: Gender, SeniorCitizen status, Partner, and Dependents.
- Services Signed Up: PhoneService, MultipleLines, InternetService, OnlineSecurity, TechSupport, StreamingTV, etc.
- Account Parameters: Tenure (months stay), Contract type, PaperlessBilling, PaymentMethod, MonthlyCharges, and TotalCharges.
- Target Variable: 'Churn' (Yes/No), indicating whether the customer terminated their service.

## 3. Data Preprocessing
Extensive data cleansing and dimensionality reduction techniques were applied to ensure optimal model performance. The 'customerID' column was removed as it held no predictive power. 'TotalCharges', initially possessing missing values represented by spaces, was coerced into numeric data types. Subsequently, the dataset was stripped of any remaining NULL/NaN values to ensure data integrity. To manage non-numeric fields, categorical variables were uniformly transformed using 'LabelEncoder', optimizing algorithmic consumption. Finally, numerical variables were normalized using a 'StandardScaler', reducing the distance scale and boosting model convergence speed.

## 4. Application of the Learning Algorithm
Four supervised classification algorithms were chosen, each serving a distinct mathematical approach to ensure robust comparability:
- Logistic Regression (Baseline): Background: A statistical model that uses a logistic function to model a binary dependent variable. Justification: Serves as a strong linear, easily interpretable baseline to measure threshold boundaries for churning probabilities.
- Decision Tree: Background: A non-parametric supervised learning method used for classification using a tree-like model of decisions. Justification: Handles non-linear relationships well and provides deep visual interpretability of feature importance (e.g., tenure length splits).
- K-Nearest Neighbors (KNN): Background: A non-parametric, lazy learning algorithm that classifies new cases based on a similarity measure. Justification: Selected for its capability to leverage proximity groupings based on billing sizes and tenure lengths.
- Random Forest: Background: An ensemble learning method that operates by constructing a multitude of decision trees at training time. Justification: Helps solve the overfitting tendencies of single decision trees and provides robust generalization for real-world unseen data.

## 5. Implementation & Results
The core logic utilizes the `scikit-learn` ecosystem with pipelines for splitting training and testing data uniformly. The metrics captured reflect model accuracy and F1 scores:
- Logistic Regression achieved a highly competitive accuracy (~80%), proving that simple linear separations exist for churning customers based on contract duration.
- Random Forest mirrored top-tier accuracy while offering excellent feature variance minimization.
- Decision Tree yielded slightly lower test accuracy (approx. 73%) compared to training accuracy, implying slight overfitting.
- KNN maintained moderate accuracy (~76%) contingent upon extensive 'K' testing and distance scaling.

## 6. Model Comparison & Best Model Selection
A comprehensive comparison of the four models reveals distinct trade-offs between predictive accuracy and model interpretability. Decision Tree achieved the lowest test accuracy (~73%) due to its inherent variance and tendency to overfit the training data. KNN performed better (~76%) but struggled with high dimensionality and varied feature distributions despite standardization. Random Forest achieved top-tier accuracy (~79-80%) by eliminating the variance of single decision trees and providing excellent robustness. However, Logistic Regression matched the highest accuracy (~80%) while taking a fraction of the computational training time.

**BEST MODEL:** Logistic Regression is definitively the best model for this specific Telco Customer Churn dataset. In a corporate environment, simply knowing 'who' will churn is only half the battle; the business must also understand 'why' they churn to take actionable preventative measures. Unlike Random Forest or KNN which act as 'black boxes', Logistic Regression provides direct, linear coefficients that explain exactly how much impact each feature (like having a 'Month-to-month contract' vs. a '2-year contract') has on the probability of a customer leaving. Given that Logistic Regression achieves identically excellent accuracy (~80%) to complex ensemble models while maintaining absolute mathematical transparency, it is the superior choice for practical business deployment.

## 7. Critical Analysis and Future Work
While excellent accuracy (~80%) was achieved, the primary limitation was dataset class imbalance (roughly 73% non-churners vs 27% churners). Future improvements must enforce synthetic sampling mechanisms such as SMOTE (Synthetic Minority Over-sampling Technique) to properly balance class distribution and improve the Recall metric specifically. Additional future work includes running Grid Search cross-validation across all models to programmatically extract the perfect hyperparameter combinations (e.g., fine-tuning max-depth for trees or n_neighbors for KNN).

## 8. Individual Contribution
The project was carried out as a collaborative effort. The responsibilities were distributed as follows:
- Gayasri Pethum Kumarana (IT22031266): Data Preprocessing Architecture & Random Forest Modeling
- Rasindu NHA (IT22092410): KNN Model Implementation
- Prarthana APS (IT22127228): Decision Tree Modeling
- Chathumadura K K K (IT22542274): Logistic Regression Modeling

*(Note: The source code is included as an appendix in the final generated PDF `report.pdf` submitted in the zip file).*
