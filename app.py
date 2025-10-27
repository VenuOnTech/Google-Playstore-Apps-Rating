import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from wordcloud import WordCloud
import warnings

# Page Configuration
st.set_page_config(
    page_title="Google Play Store Analysis",
    page_icon="📊",
    layout="wide"
)

# Data Loading and Cleaning

# Helper function for cleaning 'Size'
def clean_size(size):
    if isinstance(size, str):
        if 'M' in size:
            return float(size.replace('M', '')) * 1_000_000
        elif 'k' in size:
            return float(size.replace('k', '')) * 1_000
        elif 'Varies with device' in size:
            return np.nan
        else:
            try:
                return float(size)
            except:
                return np.nan
    return np.nan

@st.cache_data
def load_and_clean_data():
    """
    Loads, cleans, and merges the datasets.
    This function is cached to improve app performance.
    """
    # UPDATED FILE PATHS
    apps_file_path = "datasets/googleplaystore.csv"
    reviews_file_path = "datasets/googleplaystore_user_reviews.csv"
    
    try:
        df_apps = pd.read_csv(apps_file_path)
        df_reviews = pd.read_csv(reviews_file_path)
    except FileNotFoundError as e:
        st.error(f"Error loading data: {e}")
        st.error(f"Please make sure '{apps_file_path}' and '{reviews_file_path}' exist.")
        return None, None, None, None

    # Clean Apps Data
    
    # 2.1. Handle bad row and clean 'Reviews'
    df_apps['Reviews'] = pd.to_numeric(df_apps['Reviews'], errors='coerce')
    df_apps = df_apps.dropna(subset=['Reviews'])
    df_apps['Reviews'] = df_apps['Reviews'].astype(int)

    # 2.2. Clean 'Rating'
    df_apps_pre_clean = df_apps.copy() # Save pre-clean state for Suggestion 1
    df_apps['Rating'] = pd.to_numeric(df_apps['Rating'], errors='coerce')
    df_apps = df_apps[df_apps['Rating'] <= 5]
    median_rating = df_apps['Rating'].median()
    df_apps['Rating'] = df_apps['Rating'].fillna(median_rating)

    # 2.3. Clean 'Installs'
    df_apps['Installs'] = df_apps['Installs'].str.replace('+', '', regex=False)
    df_apps['Installs'] = df_apps['Installs'].str.replace(',', '', regex=False)
    df_apps['Installs'] = pd.to_numeric(df_apps['Installs'])

    # 2.4. Clean 'Size'
    df_apps['Size'] = df_apps['Size'].apply(clean_size)
    median_size = df_apps['Size'].median()
    df_apps['Size'] = df_apps['Size'].fillna(median_size)
    df_apps['Size_MB'] = df_apps['Size'] / 1_000_000

    # 2.5. Clean 'Price'
    df_apps['Price'] = df_apps['Price'].str.replace('$', '', regex=False)
    df_apps['Price'] = pd.to_numeric(df_apps['Price'])

    # 2.6. Clean 'Type' and 'Content Rating'
    df_apps['Type'] = df_apps['Type'].fillna('Free')
    df_apps['Content Rating'] = df_apps['Content Rating'].fillna('Everyone')
    
    # 2.7. Feature Engineering
    df_apps['Last_Updated_DT'] = pd.to_datetime(df_apps['Last Updated'])
    df_apps['Days_Since_Update'] = (pd.to_datetime('today') - df_apps['Last_Updated_DT']).dt.days

    # 2.8. Drop duplicates
    df_apps = df_apps.sort_values('Reviews', ascending=False)
    df_apps = df_apps.drop_duplicates(subset=['App'], keep='first')
    df_apps = df_apps.reset_index(drop=True)
    
    # Clean and Merge Reviews Data
    df_reviews_cleaned = df_reviews.dropna()
    
    # 3.A. Merge for Sentiment Polarity
    df_sentiment = df_reviews_cleaned.groupby('App')['Sentiment_Polarity'].mean().reset_index()
    df_merged = pd.merge(df_apps, df_sentiment, on='App')

    # 3.B. Merge for NLP Word Clouds
    df_reviews_text = df_reviews_cleaned.groupby('App')['Translated_Review'].apply(lambda x: ' '.join(x)).reset_index()
    df_merged_with_text = pd.merge(df_apps, df_reviews_text, on='App')
    
    return df_apps, df_merged, df_merged_with_text, df_apps_pre_clean

# --- Load Data ---
df_apps, df_merged, df_merged_with_text, df_apps_pre_clean = load_and_clean_data()

# Main App
if df_apps is not None:
    st.title("📊 Google Play Store App Analysis")
    st.markdown("This interactive dashboard presents a comprehensive analysis of the Google Play Store dataset. We explore app categories, user ratings, pricing strategies, user sentiment, and more.")

    # Set plot style
    sns.set_style('whitegrid')
    warnings.filterwarnings("ignore") # Suppress warnings in Streamlit output
    
    # Missing Data Analysis
    st.header("1. Initial Data Quality Check: Missing Data")
    st.markdown("Before cleaning, we first analyze the missing data. The heatmap shows that `Rating` is the primary column with a significant number of missing values.")
    
    fig_missing, ax_missing = plt.subplots(figsize=(12, 8))
    sns.heatmap(df_apps_pre_clean.isnull(), cbar=False, cmap='viridis', ax=ax_missing)
    ax_missing.set_title('Missing Data Heatmap (Before Cleaning)', fontsize=16)
    st.pyplot(fig_missing)

    missing_ratings_by_category = df_apps_pre_clean[df_apps_pre_clean['Rating'].isnull()]['Category'].value_counts().head()
    st.subheader("Categories with Most Missing Ratings")
    st.dataframe(missing_ratings_by_category)
    st.markdown("_This suggests that apps in categories like 'FAMILY' and 'MEDICAL' are less likely to have ratings upon initial upload. Our cleaning process fills these with the dataset's median rating._")

    # Objective 1: Category Analysis
    st.header("2. Objective 1: App Distribution and Quality by Category")
    st.markdown("Which app categories dominate the Play Store, and which are rated highest by users?")
    
    fig_cat_count, ax_cat_count = plt.subplots(figsize=(15, 10))
    sns.countplot(y='Category', data=df_apps, order=df_apps['Category'].value_counts().index, ax=ax_cat_count)
    ax_cat_count.set_title('App Distribution by Category', fontsize=16)
    ax_cat_count.set_xlabel('Number of Apps', fontsize=12)
    ax_cat_count.set_ylabel('Category', fontsize=12)
    st.pyplot(fig_cat_count)

    fig_cat_rating, ax_cat_rating = plt.subplots(figsize=(15, 10))
    category_ratings = df_apps.groupby('Category')['Rating'].mean().sort_values(ascending=False)
    sns.barplot(y=category_ratings.index, x=category_ratings.values, orient='h', ax=ax_cat_rating)
    ax_cat_rating.set_title('Average Rating by Category', fontsize=16)
    ax_cat_rating.set_xlabel('Average Rating', fontsize=12)
    ax_cat_rating.set_ylabel('Category', fontsize=12)
    ax_cat_rating.set_xlim(3.5, 5.0)
    st.pyplot(fig_cat_rating)

    # Objective 2 (Advanced): Pair Plot
    st.header("3. Objective 2: Correlations Between Key Metrics")
    st.markdown("How do key metrics relate to each other? We use a pair plot with **log-transformed** `Reviews` and `Installs` to handle the highly skewed data, revealing clearer relationships.")
    
    df_pairplot = df_apps.copy()
    df_pairplot['log_Reviews'] = np.log10(df_pairplot['Reviews'] + 1)
    df_pairplot['log_Installs'] = np.log10(df_pairplot['Installs'] + 1)
    plot_vars = ['Rating', 'Size_MB', 'Price', 'log_Reviews', 'log_Installs']
    
    g_pair = sns.pairplot(df_pairplot, vars=plot_vars, kind='reg', corner=True,
                          plot_kws={'scatter_kws': {'alpha': 0.2, 's': 10}, 'line_kws': {'color': 'red'}})
    g_pair.fig.suptitle('Pair Plot of Key Metrics (with Log-Transformed Reviews/Installs)', y=1.02, fontsize=16)
    st.pyplot(g_pair.fig)

    # Objective 3: Free vs. Paid
    st.header("4. Objective 3: Free vs. Paid App Performance")
    st.markdown("Does paying for an app guarantee a higher rating? A visual inspection with a boxplot shows paid apps have a slightly higher median rating.")
    
    fig_free_paid, ax_free_paid = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='Type', y='Rating', data=df_apps, ax=ax_free_paid)
    ax_free_paid.set_title('Rating Distribution: Free vs. Paid Apps', fontsize=16)
    ax_free_paid.set_xlabel('App Type', fontsize=12)
    ax_free_paid.set_ylabel('Rating', fontsize=12)
    st.pyplot(fig_free_paid)

    st.subheader("Statistical Test: Is the Difference Significant?")
    st.markdown("To confirm this, we use a **Mann-Whitney U test**. This non-parametric test checks if the two distributions (Free vs. Paid ratings) are statistically different.")
    
    free_ratings = df_apps[df_apps['Type'] == 'Free']['Rating']
    paid_ratings = df_apps[df_apps['Type'] == 'Paid']['Rating'].dropna()
    u_stat, p_value = stats.mannwhitneyu(free_ratings, paid_ratings, alternative='two-sided')
    
    st.code(f"""
Mann-Whitney U Test for Free vs. Paid App Ratings:
U-statistic = {u_stat:.2f}
p-value = {p_value:.4f}
    """)
    if p_value < 0.05:
        st.success("Result: The difference in ratings between Free and Paid apps is statistically significant (p < 0.05).")
    else:
        st.info("Result: The difference in ratings is NOT statistically significant (p >= 0.05).")

    # Objective 4: Sentiment vs. Rating
    st.header("5. Objective 4: User Sentiment vs. Overall Rating")
    st.markdown("Do users' written reviews (sentiment polarity) match the app's numerical star rating? We merge the two datasets to find out.")
    
    g_joint = sns.jointplot(x='Sentiment_Polarity', y='Rating', data=df_merged, kind='reg',
                          joint_kws={'scatter_kws': {'alpha': 0.3}},
                          line_kws={'color': 'red'})
    g_joint.fig.suptitle('Average Sentiment Polarity vs. App Rating', y=1.02, fontsize=16)
    st.pyplot(g_joint.fig)
    st.markdown("_As expected, there is a strong positive correlation: apps with higher average sentiment in their reviews also have higher overall star ratings._")

    # Objective 5: Content Rating
    st.header("6. Objective 5: Impact of Content Rating")
    st.markdown("How are apps distributed by content rating, and does this affect their quality?")
    
    fig_content_count, ax_content_count = plt.subplots(figsize=(10, 6))
    sns.countplot(x='Content Rating', data=df_apps, order=df_apps['Content Rating'].value_counts().index, ax=ax_content_count)
    ax_content_count.set_title('App Distribution by Content Rating', fontsize=16)
    ax_content_count.set_xlabel('Content Rating', fontsize=12)
    ax_content_count.set_ylabel('Number of Apps', fontsize=12)
    st.pyplot(fig_content_count)

    fig_content_rating, ax_content_rating = plt.subplots(figsize=(10, 7))
    sns.boxplot(x='Content Rating', y='Rating', data=df_apps, ax=ax_content_rating)
    ax_content_rating.set_title('Rating Distribution by Content Rating', fontsize=16)
    ax_content_rating.set_xlabel('Content Rating', fontsize=12)
    ax_content_rating.set_ylabel('Rating', fontsize=12)
    st.pyplot(fig_content_rating)

    # Objective 6: Date Analysis
    st.header("7. Objective 6: App Freshness (Feature Engineering)")
    st.markdown("Does updating an app frequently lead to higher ratings? We engineered a new feature, `Days_Since_Update`, to test this hypothesis.")
    
    fig_update, ax_update = plt.subplots(figsize=(10, 6))
    sns.regplot(x='Days_Since_Update', y='Rating', data=df_apps, scatter_kws={'alpha':0.2}, line_kws={'color':'red'}, ax=ax_update)
    ax_update.set_title('Rating vs. Days Since Last Update', fontsize=16)
    ax_update.set_xlabel('Days Since Last Update', fontsize=12)
    ax_update.set_ylabel('Rating', fontsize=12)
    st.pyplot(fig_update)

    correlation = df_apps[['Rating', 'Days_Since_Update']].corr().iloc[0,1]
    st.markdown(f"**Correlation: {correlation:.3f}**")
    st.markdown("_There is a slight negative correlation, suggesting that apps updated more recently (fewer days since update) tend to have slightly higher ratings._")

    # Objective 7: NLP Word Clouds
    st.header("8. Objective 7: NLP Analysis of User Reviews")
    st.markdown("What are users *saying* about the most popular apps? We generate Word Clouds from the review text of the top 3 most-reviewed apps.")
    
    top_apps = df_merged_with_text.nlargest(3, 'Reviews')['App'].tolist()

    for app_name in top_apps:
        st.subheader(f"Word Cloud for: {app_name}")
        text = df_merged_with_text[df_merged_with_text['App'] == app_name]['Translated_Review'].iloc[0]
        
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        
        fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)

else:
    st.error("Data could not be loaded. Please check the file paths and try again.")

