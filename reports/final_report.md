# FinTech Reviews Analytics: Insights and Recommendations

## Introduction

Welcome to the comprehensive analysis of Google Play Store reviews for three Ethiopian banks: Commercial Bank of Ethiopia (CBE), Bank of Abyssinia (BOA), and Dashen Bank. This project leverages advanced text preprocessing, sentiment analysis, thematic analysis, and visualizations to uncover customer satisfaction trends and recommend actionable improvements. With 1,184 reviews processed, this report delivers data-driven insights supported by seven key visualizations.

## Data Collection and Preprocessing

The journey began with scraping reviews from the Google Play Store, followed by a robust preprocessing pipeline. This included removing URLs, converting emojis to text, and applying lemmatization to ensure clean, meaningful data. Feature engineering added metrics like word counts and sentiment indicators, setting the stage for deeper analysis.

![Sentiment Trends Over Time (Monthly)](https://github.com/your-username/fintech-reviews-analytics/raw/main/data/visualizations/sentiment_trends_monthly.png)
*Figure 1: Monthly sentiment trends show fluctuating positive and negative feedback.*

## Sentiment Analysis Insights

Sentiment analysis revealed distinct patterns. Dashen Bank leads with 297 positive reviews, reflecting strong user approval. In contrast, Bank of Abyssinia recorded 226 negative reviews, indicating significant dissatisfaction. CBE falls in between with 258 positive and 125 negative reviews. These trends suggest varying user experiences across banks.

![Sentiment Counts by Bank](https://github.com/your-username/fintech-reviews-analytics/raw/main/data/visualizations/sentiment_counts.png)
*Figure 2: Bar chart of sentiment counts highlights Dashen’s positive dominance.*

## Thematic Analysis Findings

Thematic analysis identified five key themes: App Usability, Performance Issues, Transaction Problems, Customer Support, and Login Issues. Across 1,184 reviews, App Usability emerged as the dominant concern with 488 mentions. Dashen Bank faced notable Transaction Problems (60 reviews), while Abyssinia struggled with Login Issues (17 reviews).

![Theme Frequencies Across All Banks](https://github.com/your-username/fintech-reviews-analytics/raw/main/data/visualizations/theme_frequencies.png)
*Figure 3: Stacked bar chart of theme frequencies underscores App Usability’s prevalence.*

## Rating Distributions

Rating distributions vary by bank. CBE shows a median rating around 4, suggesting consistent performance. Dashen and Abyssinia exhibit lower medians with more outliers, indicating uneven user satisfaction. This variability aligns with sentiment trends, highlighting areas for improvement.

![Rating Distributions by Bank](https://github.com/your-username/fintech-reviews-analytics/raw/main/data/visualizations/rating_distributions.png)
*Figure 4: Box plot reveals CBE’s higher median rating compared to others.*

## Keyword Cloud Visualization

A keyword cloud of review terms like "good," "best," "app," and "Dashen" reflects positive feedback and app-related focus. This visualization reinforces the thematic dominance of App Usability and provides a visual summary of user language.

![Keyword Cloud](https://github.com/your-username/fintech-reviews-analytics/raw/main/data/visualizations/keyword_cloud.png)
*Figure 5: Word cloud highlights frequent positive terms.*

## Average Rating Trends

Tracking average ratings over time by bank reveals stability for CBE and fluctuations for Dashen and Abyssinia. This trend suggests CBE’s consistent user experience, while others may need targeted enhancements.

![Average Rating Trends by Bank](https://github.com/your-username/fintech-reviews-analytics/raw/main/data/visualizations/rating_trends.png)
*Figure 6: Line plot shows CBE’s stable ratings over time.*

## Keyword Correlation Heatmap

A heatmap of keyword correlations (e.g., "app" with "good," "slow" with "performance") uncovers relationships in user feedback. This analysis aids in understanding contextual usage and potential improvement areas.

![Keyword Correlation Heatmap](https://github.com/your-username/fintech-reviews-analytics/raw/main/data/visualizations/keyword_correlation.png)
*Figure 7: Heatmap illustrates keyword relationships.*

## Recommendations

Based on the analysis, here are tailored recommendations:
- **All Banks**: Add a budgeting tool to enhance App Usability.
- **CBE and Dashen**: Implement transaction recovery to address Transaction Problems.
- **Abyssinia**: Introduce crash recovery for Login Issues.
- **Dashen**: Optimize performance to reduce Transaction Problems.

## Ethical Considerations

Negative feedback may dominate due to dissatisfied users, potentially skewing theme analysis. This bias suggests a need for balanced data collection to ensure fair representation.

## Conclusion

This analysis identifies key drivers (App Usability) and pain points (Transaction Problems, Login Issues) across the banks. Supported by seven visualizations, the insights empower strategic decision-making, with ethical biases noted for future refinement.

---

*Published on Medium by [Your Name] on June 11, 2025.*