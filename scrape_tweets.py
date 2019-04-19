import os
import re
import pandas as pd
from twitter_scraper import get_tweets

if __name__ == "__main__":
    # users = ['Youtube', 'Twitter', 'instagram',
    #          'BBCBreaking', 'Reuters', 'cnnbrk', 'nytimes',
    #          'ExpressTechie', 'techreview', 'hcltech', 'NASA_Technology',
    #          'Inspire_Us', 'BuddhaQuotes', 'wordstionary',
    #          'BarackObama', 'justinbieber', 'Cristiano',
    #          'realDonaldTrump', 'BillGates', 'jimmyfallon']
    users = ['Funny_Truth', 'ohteenquotes', 'wordstionary',
             'BuddhaQuotes', 'Inspire_Us', 'FactSoup', 'MrKeyNotes1',
             'IntThings', 'NASA_Technology', 'hcltech', 'techreview']
    
    tweets = []
    for user in users:
        print(f'Scraping @{user}...')
        t_list = []
        for tweet in get_tweets(user=user, pages=30):
            tweet['user'] = user
            t_list.append(tweet)
        tweets.extend(t_list)
    
    print('Creating dataframe...')
    df = pd.DataFrame(tweets)
    df = df[['tweetId', 'time', 'user', 'text', 'likes', 'retweets', 'replies']]

    print('Saving as CSV file...')
    path = './data/'
    if not os.path.exists(path):
        os.mkdir(path)
    df.to_csv('{}{}'.format(path, 'scraped_tweets.csv'), index=False)
