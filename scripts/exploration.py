import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.api import add_constant

# Global Font & Graph Resolution
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

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

# Global Source
def add_source(fig):
    fig.subplots_adjust(bottom=0.15) 
    fig.text(0.05, 0.02, "Source: Udemy", ha='left', fontsize=9, color=colors['black'])

# Function 1: Coefficient of Variation & Skew Calculation
def coeff_var_skew(dataset, col):
    for col in col:
        # Calculate Statistics
        mean_val = dataset[col].mean()
        std_val = dataset[col].std()
        cv = std_val / mean_val
        skewness = skew(dataset[col], nan_policy='omit')  

        # Print Results for Coefficient of Variation
        print(f"\033[1m{'Column:':<15} {col}\033[0m")
        print(f"{'Mean:':<15} {mean_val:,.2f}")
        print(f"{'Std Dev:':<15} {std_val:,.2f}")
        print(f"{'CV:':<15} {cv:.2f}")
        
        # Interpret CV
        if cv > 1:
            interpretation = "The data is highly spread relative to the mean."
        elif cv > 0.5:
            interpretation = "The data has moderate spread."
        else:
            interpretation = "The data has low spread."

        print(f"{'Interpretation:':<15} {interpretation}")

        # Interpret Skewness
        if skewness > 1:
            skew_interpretation = "The data is highly right-skewed."
        elif skewness > 0.5:
            skew_interpretation = "The data is moderately right-skewed."
        elif skewness < -1:
            skew_interpretation = "The data is highly left-skewed."
        elif skewness < -0.5:
            skew_interpretation = "The data is moderately left-skewed."
        else:
            skew_interpretation = "The data is approximately symmetric."

        print(f"{'Skewness:':<15} {skewness:.2f}")
        print(f"{'Skew Interpretation:':<15} {skew_interpretation}")
        print("-" * 40)


# Function 2: Categorical Unique Value Count
def cat_unique_counts(df, cat_col):
    for i in cat_col:
        print(df[i].value_counts(dropna=False, normalize=True))
        print("*" * 40)



# Function 3: Univariate Numerical: Histplot & Box Plot Display
def num_uni_hist_box(dataset, col):
    for col in col:
        # Histplot
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=dataset, x=col, kde=True, color=colors['blue'], ax=ax, bins=20)
        ax.grid(which="major", color=colors['gray'], alpha=0.4, linestyle='-', linewidth=0.5)
        ax.spines[['top', 'right']].set_visible(False)

        # Title 1: Column Name
        ax.set_title(f"{col}", fontsize=14, weight='bold', loc='left', pad=15)

        # Title 2: Graph Type
        ax.text(0, 1.03, 'Histplot', fontsize=10, ha='left', va='center', transform=ax.transAxes, color=colors['black'])

        # Source Reference
        add_source(fig)

        plt.tight_layout()
        plt.show()

        # Boxplot
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=dataset, x=col, color=colors['blue'], ax=ax)
        ax.grid(which="major", color=colors['gray'], alpha=0.4, linestyle='-', linewidth=0.5)
        ax.spines[['top', 'right']].set_visible(False)

        # Title 1: Column Name 
        ax.set_title(f"{col}", fontsize=14, weight='bold', loc='left', pad=15)

        # Title 2: Graph Type 
        ax.text(0, 1.03, 'BoxPlot', fontsize=10, ha='left', va='center', transform=ax.transAxes, color=colors['black'])

        # Source Reference
        add_source(fig)

        plt.tight_layout()
        plt.show()

        print("*" * 40)

# Function 4: Univariate Categorical: Top Count & Barchart
def cat_uni_topcount(dataset, col):
    for col in col:
        # Get the top 5 percentages
        top_5 = dataset[col].value_counts(normalize=True).sort_values(ascending=False).head(5) * 100

        # Plot the bar chart
        fig, ax = plt.subplots(figsize=(6, 4))
        top_5.plot(kind='bar', color=colors['red'], ax=ax, edgecolor='black', alpha=0.5)
        ax.grid(which="major", color=colors['gray'], alpha=0.4, linestyle='-', linewidth=0.5)
        ax.spines[['top', 'right']].set_visible(False)

        # Title 1: Column Name 
        ax.set_title(f'Top 5 {col} by Percentage', fontsize=14, weight='bold', loc='left', pad=15)

        # Title 2: Graph Type
        ax.text(0, 1.03, 'Bar Chart', fontsize=10, ha='left', va='center', transform=ax.transAxes, color=colors['black'])

        # Rotate x-axis labels to 45 degrees
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        plt.show()

        print(f"Top 5 percentages for {col}:\n{top_5}")
        print("*" * 40)

# Function 5: Bivariate Numerical Scatterplot & Histplot with HuE
def num_biv_scat_hist_whue(dataset, col, idx_col, hue_col):
    for col in col:
        # Scatterplot
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=dataset, x=pd.to_datetime(idx_col), y=col, hue=hue_col, ax=ax, palette='Set1')
        ax.grid(which="major", color=colors['gray'], alpha=0.4, linestyle='-', linewidth=0.5)
        ax.spines[['top', 'right']].set_visible(False)

        # Title 1: Column Name
        ax.set_title(f"{col}", fontsize=14, weight='bold', loc='left', pad=15)

        # Title 2: Graph Type
        ax.text(0, 1.03, 'Scatterplot', fontsize=10, ha='left', va='center', transform=ax.transAxes, color=colors['black'])

        plt.tight_layout()
        plt.show()

        # Histplot
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data=dataset, x=col, kde=True, color=colors['blue'], hue=hue_col, ax=ax)
        ax.grid(which="major", color=colors['gray'], alpha=0.4, linestyle='-', linewidth=0.5)
        ax.spines[['top', 'right']].set_visible(False)

        # Title 1: Column Name 
        ax.set_title(f"{col}", fontsize=14, weight='bold', loc='left', pad=15)

        # Title 2: Graph Type 
        ax.text(0, 1.03, 'Histplot', fontsize=10, ha='left', va='center', transform=ax.transAxes, color=colors['black'])

        # Source Reference
        add_source(fig)

        plt.tight_layout()
        plt.show()

        print("*" * 40)


# Function 6: Text Plot Elbow Method
def text_cluster_inertia_plot(range_n_clusters, inertia):
    # Create figure 
    plt.figure(figsize=(8, 6))

    # Plotting Inertia
    plt.plot(range_n_clusters, inertia, marker='o', color='blue')

    # Styling the grid
    plt.grid(which="major", color='gray', alpha=0.4, linestyle='-', linewidth=0.5)

    # Removing top and right spines
    plt.gca().spines[['top', 'right']].set_visible(False)

    # Title 1: Main Title
    plt.title('Ad Topic Clusters', fontsize=14, weight='bold', loc='left', pad=15)

    # Title 2: Subtitle 
    plt.text(0, 1.02, 'Elbow Method', fontsize=10, ha='left', va='center', transform=plt.gca().transAxes, color='black')

    # Labeling axes
    plt.xlabel('Number of clusters', fontsize=12)
    plt.ylabel('Inertia', fontsize=12)

    # Customizing ticks
    plt.xticks(range_n_clusters, fontsize=10)
    plt.yticks(fontsize=10)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

# Function 7: PCA Plot Variance Explained
def pca_variance_plot(explained_variance, n_features):
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot Explained variance
    ax.plot(range(1, n_features), explained_variance, color='blue', marker='o', label='Explained Variance')

    # Horizontal Red Line for 90% variance
    ax.hlines(y=0.9, xmin=1, xmax=n_features, color='red', linestyle='--', linewidth=1.5, label='90% Variance')

    # Styling the grid
    ax.grid(which="major", color='gray', alpha=0.4, linestyle='-', linewidth=0.5)
    ax.spines[['top', 'right']].set_visible(False)

    # Axis labels
    ax.set_xlabel('Number of Components', fontsize=12, weight='bold', labelpad=10)
    ax.set_ylabel('Variance Explained', fontsize=12, weight='bold', labelpad=10)

    # Title 1
    ax.set_title('PCA Variance Explained ', fontsize=14, weight='bold', loc='left', pad=15)

    # Title 2
    ax.text(0, 1.03, 'Number of Components', fontsize=10, ha='left', va='center', transform=ax.transAxes, color='black')

    # Adding legend
    ax.legend(loc='lower right', fontsize=10, frameon=False)

    # Adjust layout
    plt.tight_layout()
    plt.show()

# Function 8: High Averages Display
def pca_avg_high_display(val):
    if val <= -0.40:
        return 'background-color: pink'
    elif val >= 0.40:
        return 'background-color: skyblue'

# Function 9: Variance Inflation Factor
def vif_calculation(dataframe, vif_selection):
    # Subset the dataframe to only include the vif_selection columns
    selected_data = dataframe[vif_selection]

    # Add a constant to the data (intercept term for the regression model)
    selected_data = add_constant(selected_data)

    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Feature"] = selected_data.columns
    vif_data["VIF"] = [variance_inflation_factor(selected_data.values, i) for i in range(selected_data.shape[1])]

    return vif_data

















