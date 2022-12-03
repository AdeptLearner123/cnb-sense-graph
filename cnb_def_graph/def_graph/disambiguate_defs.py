from cnb_def_graph.token_tagger.token_tagger import TokenTagger
from cnb_def_graph.sense_proposer.sense_proposer import SenseProposer
from cnb_def_graph.disambiguator.disambiguator import Disambiguator
from cnb_def_graph.utils.read_dicts import read_dicts

from tqdm import tqdm
import os
import json
from config import DISAMBIGUATION_BATCHES, DRY_RUN_SENSES
from argparse import ArgumentParser

SAVE_INTERVAL = 2
CHUNK_SIZE = 10000

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--use-amp", action="store_true")
    args = parser.parse_args()
    return args.use_amp


def get_disambiguated_sense_ids():
    sense_ids = set()

    if not os.path.exists(DISAMBIGUATION_BATCHES):
        os.mkdir(DISAMBIGUATION_BATCHES)

    for filename in os.listdir(DISAMBIGUATION_BATCHES):
        if filename.endswith(".json"):
            with open(os.path.join(DISAMBIGUATION_BATCHES, filename), "r") as file:
                file_json = json.loads(file.read())
                sense_ids.update(file_json.keys())
    return sense_ids


def save(batch_id, batch_text_senses):
    print("Saving", len(batch_text_senses))
    filename = f"batch_{batch_id}.json"
    path = os.path.join(DISAMBIGUATION_BATCHES, filename)

    if os.path.isfile(path):
        raise "Batch file path already exists: " + path

    with open(path, "w+") as file:
        file.write(json.dumps(batch_text_senses, sort_keys=True, indent=4, ensure_ascii=False))


def divide_chunks(sense_ids):
    return [sense_ids[i : i + CHUNK_SIZE] for i in range(0, len(sense_ids), CHUNK_SIZE) ]


def disambiguate_defs(dictionary, sense_ids, start_batch_id, should_save):
    use_amp = parse_args()
    
    token_tagger = TokenTagger()
    sense_proposer = SenseProposer()
    disambiguator = Disambiguator(use_amp=use_amp)

    batch_id = start_batch_id

    for chunk_sense_ids in tqdm(divide_chunks(sense_ids)):
        print("Calling")
        definition_list = [ dictionary[sense_id]["definition"] for sense_id in chunk_sense_ids ]
        token_tags_list = [ token_tagger.tokenize_tag(definition) for definition in definition_list ]
        proposals_compounds_list = [ sense_proposer.propose_senses(token_tags) for token_tags in token_tags_list ]
        token_proposals_list = [ proposals for proposals, _ in proposals_compounds_list ]
        compound_indices_list = [ compound_indices for _, compound_indices in proposals_compounds_list ]

        senses_list = disambiguator.batch_disambiguate(chunk_sense_ids, token_proposals_list, compound_indices_list)
        batch_result = { sense_id: senses for sense_id, senses in zip(chunk_sense_ids, senses_list) }

        if should_save:
            save(batch_id, batch_result)
            batch_id += CHUNK_SIZE

    """
    for i, sense_id in tqdm(list(enumerate(sense_ids))):
        definition = dictionary[sense_id]["definition"]
        token_tags = token_tagger.tokenize_tag(definition)
        token_proposals, compound_indices = sense_proposer.propose_senses(token_tags)
        senses = disambiguator.disambiguate(sense_id, token_proposals, compound_indices)
        #senses = disambiguator.batch_disambiguate([ sense_id ], [ token_proposals ], [ compound_indices ])[0]
        senses = [ sense for sense in senses if sense is not None ]
        batch_result[sense_id] = list(set(senses))

        if should_save and (i + 1) % SAVE_INTERVAL == 0:
            save(batch_id, batch_result)
            batch_result = dict()
            batch_id += SAVE_INTERVAL
    if should_save:
        save(batch_id, batch_result)    
    """


def disambiguate_all():
    dictionary = read_dicts()
    disambiguated_text_ids = get_disambiguated_sense_ids()
    missing_sense_ids = set(dictionary.keys()).difference(disambiguated_text_ids)

    print("Sense ids:", len(missing_sense_ids), "/", len(dictionary))

    disambiguate_defs(dictionary, missing_sense_ids, len(disambiguated_text_ids), True)


def dry_run():
    with open(DRY_RUN_SENSES, "r") as file:
        sense_ids = file.read().splitlines()
    
    dictionary = read_dicts()

    disambiguate_defs(dictionary, sense_ids, 0, False)


def create_dry_run():
    import random

    dictionary = read_dicts()
    dry_run_senses = random.sample(dictionary.keys(), 512)
    with open(DRY_RUN_SENSES, "w+") as file:
        file.write("\n".join(dry_run_senses))


if __name__ == "__main__":
    main()