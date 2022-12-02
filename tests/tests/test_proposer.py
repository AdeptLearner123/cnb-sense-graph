from cnb_def_graph.sense_proposer.sense_proposer import SenseProposer
from config import TEST_LABELS

import os
import json

def test_tokenizer():
    sense_proposer = SenseProposer()

    for filename in os.listdir(TEST_LABELS):
        with open(os.path.join(TEST_LABELS, filename), "r") as file:
            labels = json.loads(file.read())

            token_tags = [ (item["token"], item["tag"]) for item in labels["tokens"] ]
            expected_proposals = [ item["proposals"] if "proposals" in item else [] for item in labels["tokens"] ]
            expected_compound_indices = [ (sense, (start, end)) for sense, start, end in labels["compound_indices"] ]

            token_proposals, compound_indices = sense_proposer.propose_senses(token_tags)

            assert compound_indices == expected_compound_indices, f"Expected compound indices {expected_compound_indices} but was {compound_indices}"

            for (token, proposals), expected_proposals in zip(token_proposals, expected_proposals):
                print("Comparing", token, sorted(expected_proposals), sorted(proposals))
                assert sorted(proposals) == sorted(expected_proposals), f"Expected token {token} to have proposals {expected_proposals} but was {proposals}"