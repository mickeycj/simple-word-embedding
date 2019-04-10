import os
import re
import pandas as pd
from twitter_scraper import get_tweets

def preprocess_tweet(tweet):
    return re.sub(r'http\S+', '',
                  re.sub(r'pic.twitter\S+', '',
                         re.sub(r'@\S+', '', tweet)))

if __name__ == "__main__":
    users = ['Youtube', 'Twitter', 'instagram',
                'BBCBreaking', 'Reuters', 'cnnbrk', 'nytimes',
                'ExpressTechie', 'techreview', 'hcltech', 'NASA_Technology',
                'Inspire_Us', 'BuddhaQuotes', 'wordstionary',
                'BarackObama', 'justinbieber', 'Cristiano',
                'realDonaldTrump', 'BillGates', 'jimmyfallon']
    
    tweets = []
    for user in users:
        print(f'Scraping @{user}...')
        t_list = []
        for tweet in get_tweets(user=user):
            tweet['user'] = user
            t_list.append(tweet)
        tweets.extend(t_list)
    
    print('Creating dataframe...')
    df = pd.DataFrame(tweets)
    df['clean_text'] = df['text'].apply(preprocess_tweet)
    df = df[['tweetId', 'time', 'user', 'text', 'clean_text', 'replies', 'retweets', 'likes']]

    print('Saving as CSV file...')
    df.to_csv('./data/scraped_tweets.csv', index=False)
