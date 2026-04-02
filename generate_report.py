from fpdf import FPDF
import textwrap
import os

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)

pdf.add_page()
pdf.set_font("Helvetica", style="B", size=16)
pdf.cell(0, 10, text="Machine Learning Project Report", new_x="LMARGIN", new_y="NEXT", align="C")
pdf.set_font("Helvetica", style="I", size=12)
pdf.cell(0, 10, text="Evaluating Supervised Learning Algorithms for Customer Churn Prediction", new_x="LMARGIN", new_y="NEXT", align="C")
pdf.ln(10)

def add_heading(txt):
    pdf.set_font("Helvetica", style="B", size=13)
    pdf.cell(0, 10, text=txt, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)

def add_paragraph(txt):
    pdf.set_font("Helvetica", size=11)
    for line in textwrap.wrap(txt, width=95):
        pdf.cell(0, 6, text=line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

add_heading("1. Selection of an Appropriate Dataset")
add_paragraph("The dataset selected for this project is the Telco Customer Churn dataset. This dataset was selected because it represents a highly complex, real-world business problem faced by subscription-based services. Customer churn directly impacts company revenue, making it a critical scenario to evaluate using supervised learning. It contains a diverse mix of numerical, categorical, and binary data, moving beyond simple tutorial datasets, and is publicly hosted and widely recognized.")

add_heading("2. Description of the Dataset")
add_paragraph("Source Link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
add_paragraph("The dataset comprises 7,043 instances (customers) and 21 attributes. The context involves understanding customer behavior to prevent them from leaving. Attributes include:")
add_paragraph("- Demographic Data: Gender, SeniorCitizen status, Partner, and Dependents.")
add_paragraph("- Services Signed Up: PhoneService, MultipleLines, InternetService, OnlineSecurity, TechSupport, StreamingTV, etc.")
add_paragraph("- Account Parameters: Tenure (months stay), Contract type, PaperlessBilling, PaymentMethod, MonthlyCharges, and TotalCharges.")
add_paragraph("- Target Variable: 'Churn' (Yes/No), indicating whether the customer terminated their service.")

add_heading("3. Data Preprocessing")
add_paragraph("Extensive data cleansing and dimensionality reduction techniques were applied to ensure optimal model performance. The 'customerID' column was removed as it held no predictive power. 'TotalCharges', initially possessing missing values represented by spaces, was coerced into numeric data types. Subsequently, the dataset was stripped of any remaining NULL/NaN values to ensure data integrity. To manage non-numeric fields, categorical variables were uniformly transformed using 'LabelEncoder', optimizing algorithmic consumption. Finally, numerical variables were normalized using a 'StandardScaler', reducing the distance scale and boosting model convergence speed.")

add_heading("4. Application of the Learning Algorithm")
add_paragraph("Four supervised classification algorithms were chosen, each serving a distinct mathematical approach to ensure robust comparability:")
add_paragraph("A. Logistic Regression (Baseline): Background: A statistical model that uses a logistic function to model a binary dependent variable. Justification: Serves as a strong linear, easily interpretable baseline to measure threshold boundaries for churning probabilities.")
add_paragraph("B. Decision Tree: Background: A non-parametric supervised learning method used for classification using a tree-like model of decisions. Justification: Handles non-linear relationships well and provides deep visual interpretability of feature importance (e.g., tenure length splits).")
add_paragraph("C. K-Nearest Neighbors (KNN): Background: A non-parametric, lazy learning algorithm that classifies new cases based on a similarity measure. Justification: Selected for its capability to leverage proximity groupings based on billing sizes and tenure lengths.")
add_paragraph("D. Random Forest: Background: An ensemble learning method that operates by constructing a multitude of decision trees at training time. Justification: Helps solve the overfitting tendencies of single decision trees and provides robust generalization for real-world unseen data.")

add_heading("5. Implementation & Results")
add_paragraph("The core logic utilizes the `scikit-learn` ecosystem with pipelines for splitting training and testing data uniformly. The metrics captured reflect model accuracy and F1 scores:")
add_paragraph("- Logistic Regression achieved a highly competitive accuracy (~80%), proving that simple linear separations exist for churning customers based on contract duration.")
add_paragraph("- Random Forest mirrored top-tier accuracy while offering excellent feature variance minimization.")
add_paragraph("- Decision Tree yielded slightly lower test accuracy (approx. 73%) compared to training accuracy, implying slight overfitting.")
add_paragraph("- KNN maintained moderate accuracy (~76%) contingent upon extensive 'K' testing and distance scaling.")

add_heading("6. Model Comparison & Best Model Selection")
add_paragraph("A comprehensive comparison of the four models reveals distinct trade-offs between predictive accuracy and model interpretability. Decision Tree achieved the lowest test accuracy (~73%) due to its inherent variance and tendency to overfit the training data. KNN performed better (~76%) but struggled with high dimensionality and varied feature distributions despite standardization. Random Forest achieved top-tier accuracy (~79-80%) by eliminating the variance of single decision trees and providing excellent robustness. However, Logistic Regression matched the highest accuracy (~80%) while taking a fraction of the computational training time.")
add_paragraph("BEST MODEL: Logistic Regression is definitively the best model for this specific Telco Customer Churn dataset. In a corporate environment, simply knowing 'who' will churn is only half the battle; the business must also understand 'why' they churn to take actionable preventative measures. Unlike Random Forest or KNN which act as 'black boxes', Logistic Regression provides direct, linear coefficients that explain exactly how much impact each feature (like having a 'Month-to-month contract' vs. a '2-year contract') has on the probability of a customer leaving. Given that Logistic Regression achieves identically excellent accuracy (~80%) to complex ensemble models while maintaining absolute mathematical transparency, it is the superior choice for practical business deployment.")

add_heading("7. Critical Analysis and Future Work")
add_paragraph("While excellent accuracy (~80%) was achieved, the primary limitation was dataset class imbalance (roughly 73% non-churners vs 27% churners). Future improvements must enforce synthetic sampling mechanisms such as SMOTE (Synthetic Minority Over-sampling Technique) to properly balance class distribution and improve the Recall metric specifically. Additional future work includes running Grid Search cross-validation across all models to programmatically extract the perfect hyperparameter combinations (e.g., fine-tuning max-depth for trees or n_neighbors for KNN).")

add_heading("8. Individual Contribution")
add_paragraph("The project was carried out as a collaborative effort. The responsibilities were distributed as follows:")
add_paragraph("- Gayasri Pethum Kumarana (IT22031266): Data Preprocessing Architecture & Random Forest Modeling")
add_paragraph("- Rasindu NHA (IT22092410): KNN Model Implementation")
add_paragraph("- Prarthana APS (IT22127228): Decision Tree Modeling")
add_paragraph("- Chathumadura K K K (IT22542274): Logistic Regression Modeling")

pdf.add_page()
pdf.set_font("Helvetica", style="B", size=14)
pdf.cell(0, 10, text="Appendix: Source Code", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Courier", size=8)

source_files = [
    "src/preprocess.py", 
    "src/notebook/decision_tree/decision_tree.py",
    "src/notebook/KNN/knn.py",
    "src/notebook/logistic_regression/logistic_regression.py",
    "src/notebook/random_forest/random_forest.txt"
]

for filepath in source_files:
    if os.path.exists(filepath):
        pdf.set_font("Helvetica", style="B", size=10)
        pdf.cell(0, 10, text=f"--- {os.path.basename(filepath)} ---", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Courier", size=8)
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                clean_line = line.encode('ascii', 'ignore').decode('ascii').rstrip()
                if not clean_line.strip():
                    pdf.cell(0, 4, text="", new_x="LMARGIN", new_y="NEXT")
                else:
                    wrapped_lines = textwrap.wrap(clean_line, width=110)
                    for w_line in wrapped_lines:
                        if len(w_line) <= 200:
                            pdf.cell(0, 4, text=w_line, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)

try:
    pdf.output("report.pdf")
    print("Successfully generated report.pdf!")
except Exception as e:
    print(f"Error generating PDF: {e}")
