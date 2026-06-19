import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def save_plot(name):
    path = f"artifacts/plots/{name}.png"
    plt.savefig(path)
    plt.close()

def create_all_plots(df):

    os.makedirs("artifacts/plots", exist_ok=True)

    # Countplot
    sns.countplot(x='HeartDisease', data=df)
    plt.title('Heart Disease Distribution')
    save_plot("countplot")

    # Histogram
    sns.histplot(df['Age'], bins=20, kde=True)
    plt.title('Age Distribution')
    save_plot("histogram")

    # Boxplot
    sns.boxplot(x=df['Cholesterol'])
    plt.title('Cholesterol Boxplot')
    save_plot("boxplot")

    # Pairplot
    sns.pairplot(df[['Age','RestingBP','Cholesterol','MaxHR','Oldpeak','HeartDisease']], hue='HeartDisease')
    plt.savefig("artifacts/plots/pairplot.png")
    plt.close()

    # Heatmap
    cross_tab = pd.crosstab(df['ChestPainType'], df['HeartDisease'])
    sns.heatmap(cross_tab, annot=True, cmap='YlGnBu', fmt='d')
    plt.title('ChestPainType vs HeartDisease')
    save_plot("heatmap")

    # Density
    sns.kdeplot(data=df, x='Age', hue='HeartDisease', fill=True)
    plt.title('Age Density')
    save_plot("density")

    # Line plot
    df_sorted = df.sort_values('Age')
    sns.lineplot(x='Age', y='Cholesterol', data=df_sorted)
    plt.title('Age vs Cholesterol')
    save_plot("lineplot")

    # Pie chart
    df['HeartDisease'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Heart Disease Distribution')
    save_plot("piechart")

    # Scatter
    sns.scatterplot(x='Age', y='Cholesterol', hue='HeartDisease', data=df)
    plt.title('Scatter Plot')
    save_plot("scatter")

    # Violin
    sns.violinplot(x='HeartDisease', y='Age', data=df)
    plt.title('Violin Plot')
    save_plot("violin")