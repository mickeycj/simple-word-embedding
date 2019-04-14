import os
import re
import pandas as pd
from twitter_scraper import get_tweets

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
    df = df[['tweetId', 'time', 'user', 'text', 'likes', 'retweets', 'replies']]

    print('Saving as CSV file...')
    df.to_csv('./data/scraped_tweets.csv', index=False)
