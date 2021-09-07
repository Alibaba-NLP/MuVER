import os

WORLDS = {
    'train': [("american_football", 31929), ("doctor_who", 40281), ("fallout", 16992), ("final_fantasy", 14044), ("military", 104520), ("pro_wrestling", 10133), ("starwars", 87056), ("world_of_warcraft", 27677)],
    'valid': [("coronation_street", 17809), ("muppets", 21344), ("ice_hockey", 28684), ("elder_scrolls", 21712)],
    'test': [("forgotten_realms", 15603), ("lego", 10076), ("star_trek", 34430), ("yugioh", 10031)]
}

ENTITY_LINKING_BENCHMARK = {
    'train':['AIDA-YAGO2_train'],
    'dev': ['AIDA-YAGO2_testa'],
    'aida_test': ['AIDA-YAGO2_testb'],
    'test': ['ace2004_questions', 'AIDA-YAGO2_testb', 'aquaint_questions', 'clueweb_questions', 'msnbc_questions', 'wnedwiki_questions']
}