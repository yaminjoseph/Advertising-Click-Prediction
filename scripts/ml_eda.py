
# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score

# Color Palette
colors = {
    'red': '#D32F2F',
    'blue': '#1976D2',
    'cyan': '#0288D1',
    'green': '#388E3C',
    'yellow': '#FBC02D',
    'olive': '#8E8B2E',
    'purple': '#7B1FA2',
    'gray': '#BDBDBD',
    'black':'#000000'
}


# Function 1: Confusion Matrix Display
def confusion_matrix_display(y_test, y_pred, labels=None):
    # CM
    cm = confusion_matrix(y_test, y_pred)
    
    # Dataframe
    if labels is None:
        labels = ["Negative", "Positive"]
    
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {label}" for label in labels],
        columns=[f"Predicted {label}" for label in labels]
    )
    
    print("\033[1mConfusion Matrix:\033[0m") 
    print("-" * 40)
    print(cm_df)

# Function 2: Classification Report Display
def classification_report_display(y_test, y_pred, labels=None):
    # Classification Report
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True)
    
    # Dataframe for better readability
    if labels is None:
        labels = ["Negative", "Positive"]
    
    report_df = pd.DataFrame(report).transpose()
    
    print("\033[1mClassification Report:\033[0m")
    print("-" * 40)
    print(report_df)

# Function 3: ROC & AUC Display
def roc_curve_display(y_test, y_pred_prob, plot_title="ROC Curve"):
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color=colors['blue'], lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color=colors['gray'], linestyle='--', lw=2)  # Diagonal line (random classifier)
    
    # Customizing the plot
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=12)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=12)
    ax.set_title(plot_title, fontsize=16, weight='bold', loc='left', pad=15)
    ax.legend(loc='lower right', fontsize=12)

    # Grid customization
    ax.grid(which="major", color=colors['gray'], alpha=0.4, linestyle='-', linewidth=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    
    # Show the plot
    plt.tight_layout()
    plt.show()


# Function 4: Logist Regression Feature Evaluation: Elastic Net w CV
def lr_features_evaluation(df_processed, selected_col, target_column, test_size=0.25, random_state=42, cv_folds=5):
    # Set X/y
    X = df_processed[selected_col].drop(target_column, axis=1)
    y = df_processed[target_column].to_numpy() 
    
    # Train, Test, Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Load Scaler
    scaler = StandardScaler()
    
    # Scale Fit Data
    scaler.fit(X_train)
    
    # Scale Transform Data
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Load Model
    lr = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=1.0, max_iter=500)
    
    # Fit Model 
    lr.fit(X_train, y_train)
    
    # Predict
    y_pred = lr.predict(X_test)
    
    # Best Features: L1
    selected_features = X.columns[lr.coef_.flatten() != 0]
    
    # Coefficient Database
    coefficients = lr.coef_[0]
    features = X.columns

    # Coefficients
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': coefficients
    }).sort_values(by='Coefficient', ascending=False)

    # Cross-validation
    cv_scores = cross_val_score(lr, X_train, y_train, cv=cv_folds, scoring='accuracy')
    
    # Print Results
    print(f"\033[1mMean Cross-Validation Accuracy:\033[0m {np.mean(cv_scores):.2f}")
    print(f"\033[1mCross-Validation Accuracy Scores:\033[0m {', '.join([str(score) for score in cv_scores])}")
    print(f"\033[1mFeature Coefficients:\033[0m")
    print("-" * 40)
    print(f"\033[1mFeature\033[0m                   \033[1mCoefficient\033[0m")
    for feature, coef in zip(coef_df['Feature'], coef_df['Coefficient']):
        print(f"{feature:<25} {coef:.4f}")
    
    return {
        'selected_features': selected_features,
        'cv_scores': cv_scores,
        'mean_cv_score': np.mean(cv_scores),
        'coef_df': coef_df
    }

# Function 5: Elbow Method KNN
def plot_elbow_method_knn(X_train, y_train, X_test, y_test, k_range=30):
    test_error_rates = []

    for k in range(1, k_range + 1):
        # Initialize the KNN model
        model = KNeighborsClassifier(n_neighbors=k)
        
        # Fit the model 
        model.fit(X_train, y_train)
        
        # Predict 
        y_pred_test = model.predict(X_test)
        
        # Calculate the test error rate
        test_error = 1 - accuracy_score(y_test, y_pred_test)
        test_error_rates.append(test_error)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, k_range + 1), test_error_rates, marker='o', linestyle='-', color=colors['blue'])
    plt.xticks(range(1, k_range + 1, max(1, k_range // 10)))  
    plt.title("Elbow Method for Optimal K", fontsize=16)
    plt.xlabel("Number of Neighbors (k)", fontsize=14)
    plt.ylabel("Test Error Rate", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()




