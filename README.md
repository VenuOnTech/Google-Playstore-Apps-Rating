# 📊 Google Play Store Apps: A Comprehensive Analysis

This project performs an **in-depth exploratory data analysis (EDA)** on the **Google Play Store dataset**.  
The goal is to uncover insights into app categories, user ratings, sentiment, and the key factors that correlate with app success.

The final analysis is presented as an **interactive Streamlit web application**.

---

## 🗂️ 1. About the Datasets

This analysis utilizes **two separate datasets**, both located in the `datasets/` folder:

### **📁 datasets/googleplaystore.csv**
This dataset contains the core information for over **10,000 apps** on the Google Play Store.  
Each row represents a unique app and includes attributes such as:

- **App:** The name of the application  
- **Category:** The category the app belongs to (e.g., `GAME`, `FAMILY`, `MEDICAL`)  
- **Rating:** The average user rating for the app (out of 5)  
- **Reviews:** The total number of user reviews  
- **Size:** The size of the app (in MBs or KBs)  
- **Installs:** The approximate number of times the app has been installed  
- **Type:** Whether the app is `Free` or `Paid`  
- **Price:** The price of the app (if Paid)  
- **Content Rating:** The maturity rating (e.g., `Everyone`, `Teen`, `Mature 17+`)  
- **Genres:** The genre(s) of the app  
- **Last Updated:** The date the app was last updated  

### **📁 datasets/googleplaystore_user_reviews.csv**
This dataset contains approximately **64,000 user reviews** for various apps.  
The key columns are:

- **App:** The name of the app the review is for  
- **Translated_Review:** The text of the user review (translated to English)  
- **Sentiment:** A categorical sentiment (`Positive`, `Negative`, `Neutral`)  
- **Sentiment_Polarity:** A numerical score from **-1** (very negative) to **+1** (very positive)

We **merge these two datasets** to link quantitative app metrics with qualitative user sentiment.

---

## 🎯 2. Project Objectives

Our analysis was guided by **7 key objectives**:

1. **Analyze App Distribution and Quality:**  
   Investigate which app categories dominate the Play Store and which categories have the highest average user ratings.

2. **Investigate Key Metric Correlations:**  
   Use log-transforms to visualize the relationships between `Rating`, `Reviews`, `Installs`, and `Size`.

3. **Compare Free vs. Paid App Performance:**  
   Determine if paid apps receive significantly different ratings than free apps, and validate this using a **Mann-Whitney U test**.

4. **Correlate User Sentiment with Ratings:**  
   Verify if the sentiment expressed in written reviews (`Sentiment_Polarity`) aligns with the app's overall star rating.

5. **Assess the Impact of Content Rating:**  
   Analyze the distribution of apps by content rating and see how this impacts their average star rating.

6. **Analyze "App Freshness" (Feature Engineering):**  
   Engineer a new feature, `Days_Since_Update`, to test the hypothesis that apps updated more frequently have higher ratings.

7. **Perform Basic NLP on Reviews (Word Clouds):**  
   Generate word clouds for the most-reviewed apps to visually identify the most common themes and keywords in user feedback.

---

## 🔍 3. Key Findings

Our analysis uncovered several **key findings**:

- **Category:**  
  `FAMILY` and `GAME` are the largest app categories. However, `EVENTS`, `EDUCATION`, and `ART_AND_DESIGN` have the highest average user ratings — showing that **niche categories** often have more satisfied users.

- **Correlations:**  
  After a log-transform, we found a **strong positive correlation** between `Reviews` and `Installs`.  
  Also, a clear positive correlation exists between an app's `Rating` and its number of `Reviews`, indicating that **higher-rated apps tend to be more popular** (or vice versa).

- **Free vs. Paid:**  
  Paid apps have a **statistically significant (p < 0.05)** higher median rating than free apps.  
  While the difference is small, it’s **not due to random chance**.

- **Sentiment:**  
  There is a **strong, positive correlation** between an app's average `Sentiment_Polarity` and its `Rating`.  
  This confirms that user star ratings and written feedback are **highly aligned**.

- **Content Rating:**  
  The vast majority of apps are rated **Everyone**.  
  Interestingly, **Adults only 18+** apps had a slightly higher median rating, while **Mature 17+** had the widest range of ratings.

- **App Freshness:**  
  We found a **slight negative correlation** between `Rating` and `Days_Since_Update`.  
  This supports our hypothesis: apps that are updated more frequently (fewer days since their last update) tend to have **slightly higher ratings**.

- **NLP Findings (Word Clouds):**  
  Word clouds for popular apps like **WhatsApp** and **Facebook** revealed common keywords like *“easy to use”*, *“good”*, *“problems”*, *“ads”*, and *“please fix”* — providing a direct line of sight into **user pain points**.

---

## 🧠 4. Understandings & Conclusion

This project demonstrates a **complete data analysis pipeline**, from **data cleaning** and **feature engineering** to **advanced visualization** and **statistical testing**.

### **Key Takeaways:**
- A successful app is a **balance of many factors** — not just being free.  
- Success correlates with:
  - **User engagement** (high Reviews)
  - **Developer maintenance** (frequent updates)
  - **User satisfaction** (high Rating and positive Sentiment)

### **Data Quality Matters:**
The analysis also highlights the **critical importance of data cleaning**.  
Initial visualizations were misleading due to:
- Skewed data (`Reviews`, `Installs`)
- Missing values (`Rating`)

By addressing these through:
- **Log-transforms**
- **Median imputation**
- **Visualizing missing data first**

we were able to uncover **more accurate and reliable insights**.

---

## 🌐 5. Streamlit Web App  
  
This entire analysis is deployed as an **interactive Streamlit web application**.  
  
👉 **[View the Live Streamlit App Here](https://gooogle-playstore-apps-rating-3bvjj8nzavxvktdxftdjcb.streamlit.app/)**  


## ⚙️ How to Run Locally  
  
1. Clone this repository.  
  
2. Install the required libraries: pip install -r requirements.txt  
  
3. Ensure your project structure matches the map below (with data in the datasets folder).  
  
4. Run the app from your terminal (from the root directory, not from inside datasets): streamlit run streamlit_app.py  
  
# Project Folder Map  
  
google-play-store-analysis/  
│  
├── app.py                          # The main Python script for the Streamlit app  
├── requirements.txt                # List of required Python libraries  
├── README.md                       # This project readme file  
│  
└── datasets/  
    ├── googleplaystore.csv             # Dataset 1: App metrics  
    └── googleplaystore_user_reviews.csv  # Dataset 2: User reviews and sentiment  
  
