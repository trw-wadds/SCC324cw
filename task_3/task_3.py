from enum import Enum

""" Select a discourse segment which has a pronominal anaphor and features at least 3 preceding
    noun phrases. We assume that all words have been labelled for their part of speech and
    grammatical category. In particular, the noun phrases have been labelled as such and their
    head nouns have been also labelled for their gender and number. Design and implement an
    anaphora resolution algorithm which identifies the antecedent.

    selected phrase:
    'The man told the boy that the teacher had lost his cat.'
"""
class Part(Enum):
    S = 1           # Sentence
    VP = 2          # Verb Phrase
    NP = 3          # Noun Phrase
    N = 4           # Noun    
    V = 5           # Verb
    DET = 6         # Determiner
    C = 7           # Complementiser
    PRON = 8        # Pronoun

phrase = "the man told the boy that the teacher had lost his cat"