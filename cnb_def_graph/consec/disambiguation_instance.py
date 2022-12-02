class ConsecDisambiguationInstance:
    """
    Represents a single piece of text being disambiguated by Consec.
    Generates the Consec inputs sequentially in order of increasing polysemy.
    """

    MAX_CONTEXT_DEFS = 5

    def __init__(self, dictionary, tokenizer, sense_id, token_senses, compound_indices):
        self._dictionary = dictionary
        self._sense_id = sense_id
        self._token_senses = token_senses
        self._compound_indices = compound_indices
        self._set_disambiguation_order()
        self._current = 0
        self._tokenizer = tokenizer
        self._tokens = [ token for token, _ in token_senses ]

    def _get_definition(self, sense):
        return self._dictionary[sense]["definition"]
    
    def _set_disambiguation_order(self):
        disambiguated_senses = [None] * len(self._token_senses)

        token_polysemy = []
        for i, (_, senses) in enumerate(self._token_senses):
            if len(senses) == 1:
                disambiguated_senses[i] = senses[0]
            elif len(senses) > 1:
                token_polysemy.append((i, len(senses)))

        token_polysemy = sorted(token_polysemy, key=lambda item:item[1])
        disambiguation_order = [ i for i, _ in token_polysemy]
        self._disambiguated_senses = disambiguated_senses
        self._disambiguation_order = disambiguation_order
    
    def is_finished(self):
        return self._current >= len(self._disambiguation_order)
    
    def get_next_input(self):
        idx = self._disambiguation_order[self._current]
        _, candidate_senses = self._token_senses[idx]
        context_senses = [ (i, sense) for i, sense in enumerate(self._disambiguated_senses) if sense is not None and sense != self._sense_id ]
        context_senses = context_senses[:self.MAX_CONTEXT_DEFS]

        candidate_definitions = [ self._get_definition(sense) for sense in candidate_senses ]
        context_definitions = [ (i, self._get_definition(sense)) for i, sense in context_senses ]
        
        tokenizer_result = self._tokenizer.tokenize(self._tokens, idx, candidate_definitions, context_definitions)
        return tokenizer_result, (candidate_senses, candidate_definitions)
    
    def set_result(self, disambiguated_sense):
        idx = self._disambiguation_order[self._current]
        self._disambiguated_senses[idx] = disambiguated_sense
        self._current += 1
    
    def get_disambiguated_senses(self):
        return self._disambiguated_senses