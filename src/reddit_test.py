import praw

reddit = praw.Reddit(client_id='YOUR_CLIENT_ID',
                     client_secret='YOUR_CLIENT_SECRET',
                     user_agent='YOUR_USER_AGENT')

user = reddit.redditor('USERNAME')

for comment in user.comments.new(limit=None):
    print(comment.body)
