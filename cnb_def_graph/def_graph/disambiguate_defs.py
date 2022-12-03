from cnb_def_graph.token_tagger.token_tagger import TokenTagger
from cnb_def_graph.sense_proposer.sense_proposer import SenseProposer
from cnb_def_graph.disambiguator.disambiguator import Disambiguator
from cnb_def_graph.utils.read_dicts import read_dicts

from tqdm import tqdm
import os
import json
from config import DISAMBIGUATION_BATCHES
from argparse import ArgumentParser

SAVE_INTERVAL = 2

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


def main():
    use_amp = parse_args()
    
    dictionary = read_dicts()
    token_tagger = TokenTagger()
    sense_proposer = SenseProposer()
    disambiguator = Disambiguator(use_amp=use_amp)

    disambiguated_text_ids = get_disambiguated_sense_ids()
    #missing_sense_ids = set(dictionary.keys()).difference(disambiguated_text_ids)
    missing_sense_ids = ["m_en_gbus1184393.004", "m_en_gbus1095630.006", "m_en_gbus0275210.002", "m_en_gbus0593740.003", "Brooks_Wackerman", "m_en_gbus0009480.014", "Tales_of_Arcadia", "m_en_gbus0697540.012", "m_en_gbus0995940.005", "m_en_gbus0636170.011"]
    print("Sense ids:", len(missing_sense_ids), "/", len(dictionary))

    batch_result = dict()
    batch_id = len(disambiguated_text_ids)
    for i, sense_id in tqdm(list(enumerate(missing_sense_ids))):
        definition = dictionary[sense_id]["definition"]
        token_tags = token_tagger.tokenize_tag(definition)
        token_proposals, compound_indices = sense_proposer.propose_senses(token_tags)
        senses = disambiguator.disambiguate(sense_id, token_proposals, compound_indices)
        senses = [ sense for sense in senses if sense is not None ]
        batch_result[sense_id] = list(set(senses))

        if (i + 1) % SAVE_INTERVAL == 0:
            save(batch_id, batch_result)
            batch_result = dict()
            batch_id += SAVE_INTERVAL
    save(batch_id, batch_result)


if __name__ == "__main__":
    main()