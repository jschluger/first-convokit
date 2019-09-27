from convokit import Transformer, Corpus
import copy
from heapq import heappush, heappop

class Rank(Transformer):
    def fit(self, corpus: Corpus):
        return Transformer.fit(corpus)

    def transform(self, corpus: Corpus) -> Corpus:
        corpus = copy.deepcopy(corpus)
        for convo in corpus.iter_conversations():
            if 'rank' in convo.meta.keys():
                raise Exception('rank is already a key in this conversations meta! aborting')
            t = 0
            for id in convo._utterance_ids:
                u = corpus.get_utterance(id)
                t += len(u.text)
            convo.meta['rank'] = t
        return corpus
        
    def fit_transform(self, corpus: Corpus) -> Corpus:
        return corpus

    def order(self, corpus: Corpus):
        return sorted(list(corpus.iter_conversations()), key=lambda convo: convo.meta['rank'])
    
    def convo_length(self, corpus: Corpus, convo):
        t = 0
        for id in convo._utterance_ids:
            u = corpus.get_utterance(id)
            t += len(u.text)
        return t
    
    def rank(self, corpus: Corpus, score=None):
        if score==None:
            score = self.convo_length
        h = []
        for convo in corpus.iter_conversations():
            heappush(h, (score(corpus, convo),len(h),convo))
        while len(h) > 0:
            yield heappop(h)
        
    
