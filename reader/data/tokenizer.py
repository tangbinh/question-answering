import os
import json
import pexpect
import spacy
import unicodedata


class Tokenizer(object):
    def tokenize(self, text):
        raise NotImplementedError

    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)


class SpacyTokenizer(Tokenizer):
    """Based on https://github.com/facebookresearch/DrQA/blob/master/drqa/tokenizers/spacy_tokenizer.py"""
    def __init__(self, model='en', annotators=['lemma', 'pos', 'ner']):
        self.annotators = annotators
        disabled_components = ['parser']
        if not any([p in self.annotators for p in ['lemma', 'pos', 'ner']]):
            disabled_components.append('tagger')
        if 'ner' not in self.annotators:
            disabled_components.append('ner')
        self.nlp = spacy.load(model, disable=disabled_components)

    def tokenize(self, text):
        tokens = self.nlp.tokenizer(text.rstrip())
        if any([p in self.annotators for p in ['lemma', 'pos', 'ner']]):
            self.nlp.tagger(tokens)
        if 'ner' in self.annotators:
            self.nlp.entity(tokens)
        return {
            'text': text,
            'tokens': [Tokenizer.normalize(t.text) for t in tokens],
            'offsets': [(t.idx, t.idx + len(t)) for t in tokens],
            'pos': [t.tag_ for t in tokens] if 'pos' in self.annotators else None,
            'lemma': [t.lemma_ for t in tokens] if 'lemma' in self.annotators else None,
            'ner': [t.ent_type_ for t in tokens] if 'ner' in self.annotators else None,
        }


class CoreNLPTokenizer(Tokenizer):
    """Based on https://github.com/facebookresearch/DrQA/blob/master/drqa/tokenizers/corenlp_tokenizer.py"""
    def __init__(self, classpath=None, memory='2g', annotators=['lemma', 'pos', 'ner']):
        self.annotators = annotators
        self.memory = memory
        self.classpath = os.getenv('CLASSPATH') if classpath is None else classpath
        self._launch()

    def _launch(self):
        """Start the CoreNLP jar with pexpect."""
        annotators = ['tokenize', 'ssplit']
        if 'ner' in self.annotators:
            annotators.extend(['pos', 'lemma', 'ner'])
        elif 'lemma' in self.annotators:
            annotators.extend(['pos', 'lemma'])
        elif 'pos' in self.annotators:
            annotators.extend(['pos'])
        annotators = ','.join(annotators)
        options = ','.join(['untokenizable=noneDelete', 'invertible=true'])
        cmd = ['java', '-mx' + self.memory, '-cp', '"%s"' % self.classpath, 'edu.stanford.nlp.pipeline.StanfordCoreNLP',
               '-annotators', annotators, '-tokenize.options', options, '-outputFormat', 'json', '-prettyPrint', 'false']

        # We use pexpect to keep the subprocess alive and feed it commands
        self.corenlp = pexpect.spawn('/bin/bash', maxread=100000, timeout=60)
        self.corenlp.setecho(False)
        self.corenlp.sendline('stty -icanon')
        self.corenlp.sendline(' '.join(cmd))
        self.corenlp.delaybeforesend = 0
        self.corenlp.delayafterread = 0
        self.corenlp.expect_exact('NLP>', searchwindowsize=100)

    def tokenize(self, text):
        # Since we're feeding text to the commandline, we're waiting on seeing the NLP> prompt.
        if 'NLP>' in text:
            raise RuntimeError('Bad token (NLP>) in text!')

        # Sending q will cause the process to quit -- manually override
        if text.lower().strip() == 'q':
            token = text.strip()
            index = text.index(token)
            data = [(token, text[index:], (index, index + 1), 'NN', 'q', 'O')]
            return data

        clean_text = text.replace('\n', ' ')
        self.corenlp.sendline(clean_text.encode('utf-8'))
        self.corenlp.expect_exact('NLP>', searchwindowsize=100)

        # Skip to start of output (may have been stderr logging messages)
        output = self.corenlp.before
        start = output.find(b'{"sentences":')
        output = json.loads(output[start:].decode('utf-8'))
        tokens = [t for s in output['sentences'] for t in s['tokens']]
        brackets = {'-LRB-': '(', '-RRB-': ')', '-LSB-': '[', '-RSB-': ']', '-LCB-': '{', '-RCB-': '}'}

        return {
            'text': text,
            'tokens': [Tokenizer.normalize(brackets.get(t['word'], t['word'])) for t in tokens],
            'offsets': [(t['characterOffsetBegin'], t['characterOffsetEnd']) for t in tokens],
            'pos': [t['pos'] for t in tokens] if 'pos' in self.annotators else None,
            'lemma': [t['lemma'] for t in tokens] if 'lemma' in self.annotators else None,
            'ner': [t['ner'] for t in tokens] if 'ner' in self.annotators else None,
        }
