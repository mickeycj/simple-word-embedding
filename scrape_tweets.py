import os
import re
import pandas as pd
from twitter_scraper import get_tweets

def preprocess_tweet(tweet):
    return re.sub(r'http\S+', '',
                  re.sub(r'pic.twitter\S+', '',
                         re.sub(r'@\S+', '', tweet)))

if __name__ == "__main__":
    twitters = ['Youtube', 'Twitter', 'instagram',
                'BBCBreaking', 'Reuters', 'cnnbrk', 'nytimes',
                'ExpressTechie', 'techreview', 'hcltech', 'NASA_Technology',
                'Inspire_Us', 'BuddhaQuotes', 'wordstionary',
                'BarackObama', 'justinbieber', 'Cristiano',
                'realDonaldTrump', 'BillGates', 'jimmyfallon']
    
    tweets = []
    for twitter in twitters:
        print(f'Scraping @{twitter}...')
        tweets.extend([tweet for tweet in get_tweets(user=twitter)])
    
    print('Creating dataframe...')
    df = pd.DataFrame(tweets)
    df = df[['tweetId', 'time', 'text', 'replies', 'retweets', 'likes']]
    df['text'] = df['text'].apply(preprocess_tweet)

    print('Saving as CSV file...')
    df.to_csv('./data/scraped_tweets.csv', index=False)
