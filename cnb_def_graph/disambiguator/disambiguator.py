import torch

from config import CONSEC_MODEL_STATE
from cnb_def_graph.utils.read_dicts import read_dicts
from cnb_def_graph.consec.disambiguation_instance import ConsecDisambiguationInstance
from cnb_def_graph.consec.sense_extractor import SenseExtractor
from cnb_def_graph.consec.tokenizer import ConsecTokenizer

class Disambiguator:
    def __init__(self, debug_mode=False, use_amp=False):
        self._dictionary = read_dicts()
        self._debug_mode = debug_mode
        state_dict = torch.load(CONSEC_MODEL_STATE)
        self._sense_extractor = SenseExtractor(use_amp=use_amp)
        self._sense_extractor.load_state_dict(state_dict)
        self._sense_extractor.eval()
        self._tokenizer = ConsecTokenizer()

        if torch.cuda.is_available():
            self._sense_extractor.cuda()

    def _disambiguate_tokens(self, sense_id, token_senses, compound_indices):
        disambiguation_instance = ConsecDisambiguationInstance(self._dictionary, self._tokenizer, sense_id, token_senses, compound_indices)

        while not disambiguation_instance.is_finished():
            input, (senses, definitions) = disambiguation_instance.get_next_input()
            if torch.cuda.is_available():
                input = self._send_inputs_to_cuda(input)

            probs = self._sense_extractor.extract(*input)

            if self._debug_mode:
                sense_idxs = torch.tensor(probs).argsort(descending=True)
                for sense_idx in sense_idxs:
                    print(f"{senses[sense_idx]}:  {probs[sense_idx]} --- {definitions[sense_idx]}")

            sense_idx = torch.argmax(torch.tensor(probs))
            disambiguation_instance.set_result(senses[sense_idx])

        return disambiguation_instance.get_disambiguated_senses()

    def _send_inputs_to_cuda(self, inputs):
        (input_ids, attention_mask, token_types, relative_pos, def_mask, def_pos) = inputs

        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        token_types = token_types.cuda()
        relative_pos = relative_pos.cuda()
        def_mask = def_mask.cuda()

        return (input_ids, attention_mask, token_types, relative_pos, def_mask, def_pos)

    def disambiguate(self, sense_id, token_senses, compound_indices):
        senses = self._disambiguate_tokens(sense_id, token_senses, compound_indices)

        for sense, (start, end) in compound_indices:
            if sense in senses[start:end]:
                senses[start:end] = [sense] * (end - start)

        return senses