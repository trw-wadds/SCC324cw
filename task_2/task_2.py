from nltk.stem import *
import numpy as np
import string

""" Propose rules and write a program based on these rules which classifies a sentence as
    possible spam or not.

    Rule-Based Spam Filtering

    Assumptions:

    The input is only a single sentence;
    The term spam refers to 'any unsolicited and often irrelevant or inappropriate messages'
                (e.g. phising, advertising, spreading malware...);
    No unicode workarounds are used (such as ₤ instead of £) as this drastically
    increases the scope of the task.

    Approach:

    We use a multi-faceted approach, looking at keywords, formatting and potential links.
    A score-based system gives varying weights to different offences
    If the score passes a certain threshold, flag the sentence as spam
    Create a set of keywords and MWPs that may indicate a given sentence is spam. MWPs should
    be the principle focus as they have less ambiguity.
    Check for abnormal, attention grabbing formatting, such as all caps.
    Also use an algorithm to check for links.
    Check for excessive spacing to get around filters (e.g. 'f r e e')
    Normalise score by the length of the input, but only minorly since our input is constrained
    to a sentence. 
"""

# blacklist sets

high_punctuation_set = {                 # higher score than the below set
    "%", "£", "$", "€", "¥",
}

low_punctuation_set = {                  # baseline score of 1
    "!", "?", "*", "#", "@",
}

link_set = {                        # should only trigger as many times as there are links
    ".com", ".net", ".org", ".shop", ".pro", "http://", "https://", "www.", ".ly", "tinyurl"
}

word_set = {                        # relatively low score, focus more on MWPs (STEMMED)
    "subscrib", "congratul", "quot", "url", "fantast", "cheap", "cash", "afford", "buck",
    "deal", "loan", "trial", "prize", "bonu", "bank", "immedi", "bargain", "amaz", "exclus",
    "price", "wow", "debt", "offer", "urgent", "giveaway", "win", "credit", "free", "mortgag",
    "penc", "link", "financ", "limit", "appli", "euro", "risk-fre", "profit", "guarante",
    "pound", "member", "cent", "download", "earn", "sale", "dollar", "money",
}

mwp_set = {                         # this is the focus
    "you have been selected", "sign up", "claim your prize", "limited time offer",
    "act now", "winner announced", "get your free", "no credit check", "you won",
    "exclusive deal", "click here", "register now", "before its too late", "buy now",
    "buy today", "apply now", "apply today", "act today", "action required", "brand new",
    "exclusive deal", "hurry up", "while supplies last", "free consultation", 
    "access your account", "payment details", "confidential information", "adult content",
    "claim your reward", "cash in",
}

Stemmr = PorterStemmer()

# SCORE VALUE PLACEHOLDER: XSCOREX

# constants
UPPER_THRESHOLD = 3.0                   # How much uppercase should reach before triggering

# create reason list, score
# TESTS

test_sentences = [
    # Obvious spam
    "FREE PALESTINE!",
    "YOU HAVE BEEN SELECTED to claim your prize!",
    "Act now! Limited time offer!!!",
    "Click here to win a brand new car!",
    "Get your free consultation today!",
    "No credit check loan available now!!!",
    "Congratulations! You won $1,000,000!",
    "BUY NOW and get 50% off!!!",
    "Register now to access your account.",
    "Exclusive deal just for you!",
    "Payment details required to claim your reward.",
    
    # Spam with links
    "Visit http://bit.ly/offer to claim your prize!",
    "Check out www.amazing-deals.com for free gifts!",
    "Limited offer at https://cheapstuff.shop today!",
    
    # Borderline / suspicious spam
    "This is your final chance to act today.",
    "Hurry up, while supplies last!",
    "You won't believe this amazing bargain.",
    "Earn money quickly and risk-free.",
    "Download this free trial now!",
    "Exclusive bonus available if you apply today.",
    
    # Normal / non-spam
    "I will meet you at the cafe tomorrow.",
    "Can you send me the report by Monday?",
    "Let's have lunch together next week.",
    "I am excited about the new project.",
    "The weather is nice today.",
    "My cat loves playing with toys.",
    "Happy birthday! Hope you have a great day.",
    "Please review the attached document.",
    "We should schedule a meeting to discuss plans.",
    
    # Edge cases: uppercase but not spam
    "NASA LAUNCHED A NEW SATELLITE TODAY",
    "IMPORTANT: Meeting rescheduled to 3 PM",
    
    # Edge cases: punctuation heavy but normal
    "Wow!!! That was an amazing movie!!!",
    "Are you coming??? I can't wait!!!",
    
    # Mixed content
    "Free cookies at the office tomorrow!",
    "Congratulations on your promotion!",
    "Act now to save on your taxes!"  # borderline spam
]

for input in test_sentences:

    # input = "YOU HAVE BEEN SELECTED to claim your prize!"

    reasons = []
    score = 0

    # check for uppercase
    # if excessive uppercase, make flag to check for individual uppercase false
    upper_count = 0
    for c in input:
        if c.isupper():
            upper_count += 1
    if upper_count >= len(input) / UPPER_THRESHOLD: # MAYBE MAKE VARIABLE ie PUNISH 100% MORE THAN 33%
        score += 5 # XSCOREX
        reasons.append("Excessive uppercase")

    # make all lowercase
    input_lower = input.lower()

    # check for weird punctuation
    for str in link_set:                                                    # TODO: CHANGE TO REGEX
        if str in input_lower:
            score += 5 # XSCOREX
            reasons.append("Link detected")

    low_punc_count = 0
    for c in input:
        if c in high_punctuation_set:
            score += 3 # XSCOREX
            reasons.append(f"Unusual punctuation: {c}")
        elif c in low_punctuation_set:
            low_punc_count += 1 # XSCOREX
    if low_punc_count >= 3: # MAYBE NORMALISE BY SENTENCE LENGTH???????
        score += 3
        reasons.append(f"Excessive punctuation")

    # Remove punctuation
    translator = str.maketrans("", "", string.punctuation)
    input, input_lower = input.translate(translator), input_lower.translate(translator)
    input, input_lower = " " + input + " ", " " + input_lower + " "

    for mwp in mwp_set:
        if f" {mwp} " in input_lower:
            score += 5 # XSCOREX
            reasons.append(f"Key MWP detected: {mwp}")
        # Additionally, add further points if fully capitalised
        if mwp.upper() in input:
            score += 5 # XSCOREX
            reasons[-1] += " (UPPERCASE)"

    split, split_lower = input.split(" "), input_lower.split(" ")

    for w in split:
        if Stemmr.stem(w, to_lowercase=True) in word_set:
            score += 3
            reasons.append(f"Keyword detected: {w.lower()}")
            if w.isupper():
                score += 2
                reasons[-1] += " (UPPERCASE)"

    # normalise
    denom = np.sqrt(max(len(split), 3))
    normalised_score = score / denom

    # output
    # if normalised_score >= 3:
    #     print(f"The input has been flagged as spam with score: {normalised_score}. Reasons: {reasons}")
    # else:
    #     print(f"The input is likely not spam.")

    # TEST output
    if normalised_score >= 3:
        print(f"{input} -> SPAM\t\t {', '.join(reasons)}\n")
    else:
        print(f"{input} -> NOT SPAM\n")


# TESTING




""" possible test cases:
    FREE PALESTINE!
"""
