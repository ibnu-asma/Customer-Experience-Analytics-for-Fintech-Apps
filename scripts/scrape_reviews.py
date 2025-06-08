from google_play_scraper import app, Sort, reviews
import pandas as pd
from datetime import datetime

# Define app IDs (replace with correct ones if different)
apps = {
    "Commercial Bank of Ethiopia": "com.combanketh.mobilebanking",
    "Bank of Abyssinia": "com.boa.boaMobileBanking",  # Example; verify on Play Store
    "Dashen Bank": "com.dashen.dashensuperapp"  # Example; verify on Play Store
}

# Function to scrape reviews
def scrape_reviews(app_name, app_id, num_reviews=400):
    print(f"Scraping {num_reviews} reviews for {app_name}...")
    all_reviews = []
    review_count = 0
    continuation_token = None
    while review_count < num_reviews:
        batch, continuation_token = reviews(
            app_id,
            lang='en',  # English reviews
            country='et',  # Ethiopia
            sort=Sort.MOST_RELEVANT,  # Can change to NEWEST
            count=min(100, num_reviews - review_count),  # Max 100 per request
            continuation_token=continuation_token
        )
        all_reviews.extend(batch)
        review_count += len(batch)
        if not continuation_token:
            break
        # Optional: Add a delay to avoid rate limiting
        # import time; time.sleep(2)
    
    # Process reviews, handling 'at' as datetime or timestamp
    processed_reviews = []
    for r in all_reviews[:num_reviews]:
        if isinstance(r['at'], datetime):
            review_date = r['at'].strftime('%Y-%m-%d')
        else:
            review_date = datetime.fromtimestamp(r['at']).strftime('%Y-%m-%d')
        processed_reviews.append({
            "review": r['content'],
            "rating": r['score'],
            "date": review_date,
            "bank": app_name,
            "source": "Google Play"
        })
    return processed_reviews

# Scrape reviews for each bank
all_data = []
for app_name, app_id in apps.items():
    reviews_data = scrape_reviews(app_name, app_id, 400)
    all_data.extend(reviews_data)

# Convert to DataFrame and save
df = pd.DataFrame(all_data)
df.to_csv("../Data/raw_reviews.csv", index=False)
print(f"Scraped {len(df)} reviews total.")