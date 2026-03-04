from nltk.stem import *
import numpy as np
import string

""" Word-Sense Disambiguation

    Polysemous word: BOW

    Given meanings:
    
    a part of a boat;
    polite bending gesture;
    ranged historical weapon;
    an article of clothing.
    
    Assumptions:

    Input is grammatically correct English text;
    Disambiguation only uses context from the same line
                (ie. .txt files with only one word per line are unsuitable).
    
    Approach:

    We use a rule-based approach, defining sets of colours pertaining to each possible meaning.
    For each instance of the word BOW, we create window of size WINDOW_SIZE around it. For each
    word in the window, once processed, check it against the word sets. The most frequent meaning
    in the window is its estimated meaning.
    This approach is limited primarily by the innately insufficient set sizes, due to the nature of
    a modern language such as English. For instance, wikipedia has 32 listings of BOW dissambiguations.

    A sample test case is provided in test_cases_task_1.txt

"""

# Array of lists, containing: Associated word sets (0); meaning (1); current tally (2) 
Associations = [[
    {
    "captain", "marina", "hull", "rudder", "ferri", "bridg", "cargo", "deck",
    "yacht", "harbor", "moor", "pier", "mast", "coast", "keel", "crew", "sail",
    "starboard", "buoy", "helm", "port", "anchor", "dock", "stern", "voyag", "tide",
    },
    "boat", 0],
[
    {
    "perform", "kneel", "stage", "polit", "queen", "respect", "greet",
    "audienc", "acknowledg", "conclud", "worship", "applaus", "custom",
    "introduc", "tradit", "king", "templ", "ceremoni", "submiss",
    "gratitud", "honor", "princ", "apolog", "gestur", "princess",
    },
    "gesture", 0],
[ 
    {
    "quiver", "releas", "mediev", "bullsey", "compound", "string", "shaft",
    "crossbow", "forest", "recurv", "tournament", "hunt", "draw", "target",
    "arrow", "longbow", "rang", "fletch", "warrior", "archer", "accuraci",
    "battl", "aim", "practic",
    },
    "weapon", 0],
[
    {
    "costum", "fashion", "decor", "accessori", "tie", "collar", "satin", "wrap",
    "gift", "dress", "hair", "blous", "shoelac", "lace", "ribbon", "wed", "clip",
    "pink", "silk", "formal", "outfit", "neck", "packag", "headband",
    },
    "article of clothing", 0],
]

WINDOW_SIZE = 5

Stemmr = PorterStemmer()

with open("test_cases_task_1.txt","r") as file:
    for line_idx, line in enumerate(file):

        # Remove punctuation
        translator = str.maketrans("", "", string.punctuation)
        clean_line = line.translate(translator)

        split = clean_line.split(" ")
        for idx, target in enumerate(split):
            if target.lower() != "bow":
                continue

            window = split[max(0, idx-WINDOW_SIZE):min(idx+WINDOW_SIZE, len(split))] # Create window of size 5 (either side)

            # Check if each word in the window is contained in any of the word sets, increment tally if so
            for w in window:
                for i in range(4):
                    if Stemmr.stem(w, to_lowercase=True) in Associations[i][0]:
                        Associations[i][2] += 1
                        break # No duplicates in word sets, so we can stop when the word is found in one set

            tallies = [category[2] for category in Associations]
            m = max(tallies)

            if m == 0:
                result = "NONE"
            elif tallies.count(m) > 1:
                result = "AMBIGUOUS"
            else:
                result = Associations[np.argmax(tallies)][1]

            print(f"The instance of bow at line {line_idx}, index {idx} likely pertains to subject: {result.upper()}")

            # reset tallies after each window
            for i in range(len(Associations)):
                Associations[i][2] = 0