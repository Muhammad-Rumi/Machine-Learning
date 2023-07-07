from collections import Counter
from math import log10

def calculate_icity(data,vocab,key= 'spam'):
# Count the occurrences of each word in spam emails
    word_counts_am = Counter(word for sentence in data[key] for word in sentence.split())

    # Calculate the spamicity for each word
    total_am = sum(word_counts_am.values())
    dict_icity = {w.lower(): (word_counts_am.get(w,0)+1) / (total_am+ len(vocab)) for w in vocab}

    return dict_icity

def classify(message):
    p_spam_given_message = S
    p_ham_given_message = H

    for word in message.split():

        p_spam_given_message *= log10(spamicity[word]) #P(S|W)
        p_ham_given_message *= log10(hamicity[word])# P(¬S|W)

    if p_ham_given_message > p_spam_given_message:

        return 'ham'
    
    elif p_ham_given_message < p_spam_given_message:

        return 'spam'
    else:

        return 'needs human classification'

V = ['secret', 'offer', 'low', 'price', 'valued', 'customer', 'today', 'dollar','million', 'sports', 'is', 'for', 'play', 'healthy', 'pizza']

train_emails = {'spam':['million dollar offer for today', 'secret offer today', 'secret is secret' ],'ham': ['low price for valued customer', 'play secrets sports today', 'sports is healthy', 'low price pizza']}

hamicity = calculate_icity(train_emails,V,key='ham') # P(W|¬S)
spamicity = calculate_icity(train_emails,V,key='spam')# P(W|S)

S = len(train_emails['spam'])/(len(train_emails['spam']) + len(train_emails['ham'])) #P(S)
H = len(train_emails['ham'])/(len(train_emails['spam']) + len(train_emails['ham'])) # P(¬S)

print(classify('today is secret'))