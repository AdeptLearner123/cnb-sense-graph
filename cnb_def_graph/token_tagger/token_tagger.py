import spacy
from spacy.matcher import Matcher

class TokenTagger():
    SPACY_POS_TO_TAG = {
        "NOUN": "noun",
        "PROPN": "noun",
        "VERB": "verb",
        "ADJ": "adjective",
        "NUM": "numeral"
    }

    MANUAL_REMOVE_STOP_WORDS = {
        "us" # Don't want "US" as in United States to be classified as a stop word,
    }
    
    def __init__(self):
        self._nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self._nlp.Defaults.stop_words -= self.MANUAL_REMOVE_STOP_WORDS
        print(self._nlp.Defaults.stop_words)

    def _merge_proper_chunks(self, doc):
        pattern = [
            {"TEXT": { "IN": [ "The", "An", "A" ] }, "OP": "?"},
            {"POS": "PROPN", "OP": "+"},
            {"TEXT": { "IN": [ "'s", "-" ] }, "OP": "?"},
            {"POS": "PROPN", "OP": "*"}]
        
        matcher = Matcher(self._nlp.vocab)
        matcher.add("ProperChunks", [ pattern ])
        matches = matcher(doc)

        spans = [doc[start:end] for _, start, end in matches]
        spans = spacy.util.filter_spans(spans)

        with doc.retokenize() as retokenizer:
            for span in spans:
                retokenizer.merge(doc[span[0].i:span[-1].i + 1])

    def _get_token_tag(self, token):
        if token.is_stop:
            return None

        if token.pos_ in self.SPACY_POS_TO_TAG:
            return self.SPACY_POS_TO_TAG[token.pos_]
        
        return None

    def tokenize_tag(self, text):
        doc = self._nlp(text)
        self._merge_proper_chunks(doc)
        tokens = [ token.text for token in doc ]
        tags = [ self._get_token_tag(token) for token in doc ]
        return list(zip(tokens, tags))