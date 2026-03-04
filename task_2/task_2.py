import numpy as np

""" Propose rules and write a program based on these rules which classifies a sentence as
    possible spam or not.

    Rule-Based Spam Filtering

    Approach:

    We use a multi-faceted approach, looking at keywords, formatting and potential links.
    A score-based system gives varying weights to different offences
    If the score passes a certain threshold, flag the sentence as spam
    Create a set of keywords and MWPs that may indicate a given sentence is spam.
    Check for abnormal, attention grabbing formatting, such as all caps.
    Also use an algorithm to check for links
"""

# blacklist
word_set = {
    "link", "free", ""
}

mwp_set = {
    "you have been selected",
}

# possible whitelist?????

# do we want to do words and MWPs in separate passes or the same?????????????

# check for uppercase
# make all lowercase

# check for weird punctuation
# remove punctuation

# stem

# perform word matching

# 