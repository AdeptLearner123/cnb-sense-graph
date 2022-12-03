from cnb_def_graph.disambiguator.disambiguator import Disambiguator
from cnb_def_graph.sense_proposer.sense_proposer import SenseProposer
from config import TEST_LABELS

import os
import json

def test_disambiguator():
    disambiguator = Disambiguator()

    for filename in os.listdir(TEST_LABELS):
        with open(os.path.join(TEST_LABELS, filename), "r") as file:
            print("Testing", filename)
            labels = json.loads(file.read())

            sense_id = labels["sense_id"]
            token_proposals = [ (item["token"], item["proposals"] if "proposals" in item else []) for item in labels["tokens"] ]
            compound_indices = [ (sense, (start, end)) for sense, start, end in labels["compound_indices"] ]
            tokens = [ token for token, _ in token_proposals ]

            senses_list = disambiguator.batch_disambiguate([ sense_id ], [ token_proposals ], [ compound_indices ])
            senses = senses_list[0]

            expected_senses = [ item["sense"] if "sense" in item else None for item in labels["tokens"] ]
            print(expected_senses)
            print(token_proposals)
            print(senses)
            for token, sense, expected_sense_options in zip(tokens, senses, expected_senses):
                if expected_sense_options is None:
                    assert sense is None, f"Expected {token} to have no sense but was {sense}"
                else:
                    assert sense in expected_sense_options, f"Expected {token} to have sense in {expected_sense_options} but was {sense}"