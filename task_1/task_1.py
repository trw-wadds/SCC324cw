from nltk.stem import *
import numpy as np

""" Choose a polysemous word. Design and implement a simple rule-based algorithm to perform
    word-sense disambiguation for this specific word. 
    
    Polysemous word: BOW
    Given meanings:
    
    A part of a boat
    Polite bending gesture
    Ranged historical weapon
    An article of clothing
    
    Assumptions:
    Input is grammatically correct English text

    
    """

### RESET TALLIES TO 0


# 
# Array of Tuples, containing: Associated word sets (0); meaning (1); current tally (2) 
Associations = [[{
    "captain", "marina", "hull", "rudder", "ferri", "bridg", "cargo", "deck",
    "yacht", "harbor", "moor", "pier", "mast", "coast", "keel", "crew", "sail",
    "starboard", "buoy", "helm", "port", "anchor", "dock", "stern", "voyag", "tide",
    },"boat", 0],
[
    {
    "perform", "kneel", "stage", "polit", "queen", "respect", "greet", "princess",
    "audienc", "present", "acknowledg", "conclud", "worship", "applaus", "custom",
    "formal", "introduc", "tradit", "king", "templ", "ceremoni", "submiss", "gestur",
    "gratitud", "honor", "princ", "apolog",
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
    "pink", "present", "silk", "formal", "outfit", "neck", "packag", "headband",
    },
    "article",0],
]

Stemmr = PorterStemmer()

with open("test_cases_task_1.txt","r") as file:
    for line in file:
        split = line.split(" ")
        for idx, target in enumerate(split):
            if target.lower() != "bow":
                continue
            window = split[max(0, idx-5):min(idx+5, len(split))]
            for w in window:
                if Stemmr.stem(w, to_lowercase=True) in Associations[0][0]:
                    boatT+=1
                elif Stemmr.stem(w, to_lowercase=True) in Associations[1][0]:
                    ben_doverT+=1
                elif Stemmr.stem(w, to_lowercase=True) in Associations[2][0]:
                    weaponT+=1
                elif Stemmr.stem(w, to_lowercase=True) in Associations[3][0]:
                    articleT+=1
                for i in Associations[:,2]:
                    Associations[:,2][i] = 0

            print(f"word at index {idx} is likely to do with {Associations[:][1][np.argmax(Associations)]}")

            