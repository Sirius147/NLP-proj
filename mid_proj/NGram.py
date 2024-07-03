from nltk import download
from nltk.tokenize import WhitespaceTokenizer
from collections import Counter
from datasets import load_dataset
from tqdm.auto import tqdm
from numpy import log, zeros
from numpy.random import choice, seed
from math import isclose
from pprint import pprint


class NGram:

    def __init__(self, min_occur=5):
        self.vocab = {"<unk>": 0, "<s>": 1, "</s>": 2}
        self.unigram_prob = None
        self.bigram_prob = None
        self.bigram_prob_smoothed = None 
        self.min_occur = min_occur
        self.tokenizer = WhitespaceTokenizer()


    def sent_tokenize(self, sentence, use_vocab=True):
        #
        # Tokenize the input sentence 
        #
        # - sentence: input sentence
        # - use_vocab (True or False): if set, replace OOV tokens with <UNK>
        #
        sentence = sentence.lower()
        tokenized = ['<s>'] + self.tokenizer.tokenize(sentence) + ['</s>']
        if use_vocab:
            # UPDATE ZONE - START
            for item in tokenized:
                if item not in self.vocab:
                    tokenized.insert(tokenized.index(item),'<unk>') # insert unk at item idx
                    tokenized.remove(item)
            # UPDATE ZONE - END

        return tokenized

    def doc_tokenize(self, document, train=False):
        # 
        # Tokenize the input corpus, and construct the vocabulary (optional)
        # Returns the tokenized input corpus
        #
        # - document: corpus to be tokenized
        # - train: if set, construct the vocabulary from the train corpus
        # 
        tokenized_sents = []
        if train:
            for text in document:
                for sentence in text.split(' . '):
                    if sentence != '\n':  # end of sentence
                        # UPDATE ZONE - START
                        vocabs=self.sent_tokenize(sentence,use_vocab=False)
                        tokenized_sents.append(vocabs)
            voc_list=[voc for sent in tokenized_sents for voc in sent]
            counter=Counter(voc_list).most_common()
            for key,val in counter:
                if (key not in  self.vocab) and (val >= self.min_occur):
                    self.vocab[key]=len(self.vocab)   
            # min_occur 미만의 token 삭제
            return self.doc_tokenize(document,train=False)
            # UPDATE ZONE - END

        else:
            if len(self.vocab) <= 3:
                print("Error Case - Train corpus should be processed in advance.")
                exit()

            for text in document:
                for sentence in text.split(' . '):
                    if sentence != '\n':  # 문장 끝
                        # UPDATE ZONE - START
                        vocabs=self.sent_tokenize(sentence,use_vocab=True)
                        tokenized_sents.append(vocabs)   
                        # UPDATE ZONE - END
            
        return tokenized_sents

    def train(self, corpus):
        #
        # Tokenize the input corpus and estimate the uni- and bi-gram probability
        #
        # - corpus: input corpus
        #
        tokenized_sentences = self.doc_tokenize(corpus, train=True)
        print("(train) 0 - Print tokenized sentence samples")
        print(tokenized_sentences[0:20:5])
        print()
    
        vocab_size = len(self.vocab)
        # UPDATE ZONE - START
        self.unigram_prob = zeros(vocab_size) # vocab_size 크기의 1차원 array 생성
        self.bigram_prob = zeros((vocab_size, vocab_size)) # OR dict()
        self.bigram_prob_smoothed = zeros((vocab_size, vocab_size)) # OR dict()
        '''self.bigram_prob = {j:[0 for _ in self.vocab] for j in self.vocab}
        self.bigram_prob_smoothed = {j:[0 for _ in self.vocab] for j in self.vocab}'''
        # UPDATE ZONE - END
        
        print("(train) 1 - Count frequency ")
        # UPDATE ZONE - START
        full_tok_cnt=0  # min_occur 이상의 token 개수 구하기 -> unigram 계산용
        for i in tokenized_sentences:
            full_tok_cnt+=len(i)
        #build unigram,bigram count
        for sent in tqdm(tokenized_sentences):
            for tok in sent:
                self.unigram_prob[self.vocab[tok]]+=1
                # full_tok_cnt+=1
            for i in range(len(sent)-1):
                self.bigram_prob[self.vocab[sent[i]]][self.vocab[sent[i+1]]]+=1
        #build bigram smoothe count
        self.bigram_prob_smoothed=self.bigram_prob+1   
        # UPDATE ZONE - END

        print("(train) 2 - Estimate probability ")
        # UPDATE ZONE - START
        # estimate bigram prob & smoothed bigram prob
        for idx in range(vocab_size):
            for i in range(vocab_size):
                self.bigram_prob[idx][i]/=self.unigram_prob[idx]
                self.bigram_prob_smoothed[idx][i]/=(self.unigram_prob[idx] + vocab_size)        
        # estimate unigram prob
        for idx in range(vocab_size):
            self.unigram_prob[idx]/=full_tok_cnt
        # bigram check if has problem     
        for idx in range(vocab_size):
            if idx == 2: continue
            elif(isclose(self.bigram_prob[idx].sum(),1) and isclose(self.bigram_prob_smoothed[idx].sum(),1)):
                continue
            else: 
                print("bigram row's sum isn't 1")
                print(f"bigram sum:{self.bigram_prob[idx].sum()} smoothed bigram sum:{self.bigram_prob_smoothed[idx].sum()}) index:{idx}")
                exit()
        # UPDATE ZONE - END

    def get_unigram_probablity(self, tok):
        tok = tok.lower()
        # UPDATE ZONE - START
        return self.unigram_prob[self.vocab[tok]] 
        # UPDATE ZONE - END
        
    def get_bigram_probablity(self, first, second, smoothing=False):
        first, second = first.lower(), second.lower()
        # UPDATE ZONE - START
        if smoothing:
            return self.bigram_prob_smoothed[self.vocab[first]][self.vocab[second]] 
        return self.bigram_prob[self.vocab[first]][self.vocab[second]]
        # UPDATE ZONE - END
        
    def get_bigram_probability_with_context(self, last_token, smoothing=False):
        last_token = last_token.lower()
        # UPDATE ZONE - START
        if smoothing:
            return self.bigram_prob_smoothed[self.vocab[last_token]]
        return self.bigram_prob[self.vocab[last_token]]
        # UPDATE ZONE - END
        
    def calc_sent_prob(self, sentence, bigram=False, use_log=False, smoothing=False):
        # 
        # Estimate the probability of input sentence
        #
        # - bigram: if set, use bigram probability
        # - use_log: if set, use log for the probability transformation
        # - smoothing: if set, use add-1 smoothing
        #
        tokenized = self.sent_tokenize(sentence)
        # UPDATE ZONE - START
        if use_log:
            prob=0
            if bigram:
                if smoothing:
                    for i in range(len(tokenized)-1):
                        prob+=log(self.get_bigram_probability_with_context(tokenized[i],smoothing=True)[self.vocab[tokenized[i+1]]])
                else:
                    for i in range(len(tokenized)-1):
                        prob+=log(self.get_bigram_probability_with_context(tokenized[i])[self.vocab[tokenized[i+1]]])
            else:
                for i in range(1,len(tokenized)-1):
                        prob+=log(self.unigram_prob[self.vocab[tokenized[i]]])
        else:
            prob=1
            if bigram:
                if smoothing:
                    for i in range(len(tokenized)-1):
                        prob*=self.get_bigram_probability_with_context(tokenized[i],smoothing=True)[self.vocab[tokenized[i+1]]]
                else:
                    for i in range(len(tokenized)-1):
                        prob*=self.get_bigram_probability_with_context(tokenized[i])[self.vocab[tokenized[i+1]]]
            else:
                for i in range(1,len(tokenized)-1):
                        prob*=self.unigram_prob[self.vocab[tokenized[i]]]
        return prob  
        # UPDATE ZONE - END
        
    def generate(self, context, max_iter, bigram=False, smoothing=False):
        # 
        # Complete the sentence generation
        # Use the bigram probability for generation
        # Sample the next token according to the corresponding distribution given by the context
        # If the end-of-sentence token (</s>) is generated, quit.
        #
        # - context: a word to be used for bigram prob estimation
        # - max_iter: the maximum number of iterations. 
        # - bigram: if set, use bigram probability
        #
        seed(len(context) * max_iter)
        inverse_vocab = {idx: tok for tok, idx in self.vocab.items()}
        ordered_tokens = [inverse_vocab[idx] for idx in range(len(self.vocab))]
        
        # UPDATE ZONE - START
        # choice(self.vocab.keys(),max_iter,p=self.bigram_prob_smoothed[last_voc])
        if bigram:
            last_voc=context.split(' ')[-1]
            if smoothing:
                for _ in range(max_iter):
                    next=choice(ordered_tokens,1,p=self.get_bigram_probability_with_context(last_voc,smoothing=True))[0]
                    if next =='</s>':
                        context+=(' '+next)
                        break
                    else:
                        context+=(' '+next)
                        last_voc=next
            else:
                for _ in range(max_iter):
                    next=choice(ordered_tokens,1,p=self.get_bigram_probability_with_context(last_voc))[0]
                    if next =='</s>':
                        context+=(' '+next)
                        break
                    else:
                        context+=(' '+next)
                        last_voc=next
        else:
            for _ in range(max_iter):
                next=choice(ordered_tokens,1,p=self.unigram_prob)[0]
                if next=='</s>':
                    context+=(' '+next)
                    break
                else:
                    context+=(' '+next)

        return context
        # UPDATE ZONE - END
    

if __name__ == "__main__":

    train_corpus = [text for text in load_dataset("wikitext", 'wikitext-103-v1')['train']['text'] if text and not text.startswith(' =')][:10000]
    download('punkt')

    print(" 1. Train ")
    ngram_model = NGram()
    ngram_model.train(train_corpus)
    print()

    print("unigram prob samples (index 0 to 9)")
    pprint(ngram_model.unigram_prob[0:10])
    print("bigram prob samples (index 0 to 9)")
    pprint(ngram_model.bigram_prob[0:10][0:10])

    print(" 2. Test ")
    print()
    print(" 2-1. N-gram probability")
    print(f' Prob("I"): {ngram_model.get_unigram_probablity("I"):.6f}')
    print(f' Prob("want"): {ngram_model.get_unigram_probablity("want"):.6f}')
    print(f' Prob("am"): {ngram_model.get_unigram_probablity("am"):.6f}')

    print(f' Unsmoothed Prob("I want"): {ngram_model.get_bigram_probablity("I", "want", smoothing=False):.6f}')
    print(f' Unsmoothed Prob("I am"): {ngram_model.get_bigram_probablity("I", "am", smoothing=False):.6f}')

    print(f' Smoothed Prob("I want"): {ngram_model.get_bigram_probablity("I", "want", smoothing=True):.6f}')
    print(f' Smoothed Prob("I am"): {ngram_model.get_bigram_probablity("I", "am", smoothing=True):.6f}')
    print()

    print(" 2-2. Sentence probability")
    sentence = "It was Second Europan War."
    print(f" Target: {sentence}")

    print()
    sentence_unigram_prob = ngram_model.calc_sent_prob(sentence, bigram=False, use_log=False)
    print(f" Prob (unigram): {sentence_unigram_prob:.6f}")

    sentence_unigram_logprob = ngram_model.calc_sent_prob(sentence, bigram=False, use_log=True)
    print(f" Log Prob (unigram): {sentence_unigram_logprob:.6f}")

    sentence_bigram_prob_unsmoothed = ngram_model.calc_sent_prob(sentence, bigram=True, use_log=True, smoothing=False)
    print(f" Log Prob (bigram unsmoothed): {sentence_bigram_prob_unsmoothed:.6f}")

    sentence_bigram_prob_smoothed = ngram_model.calc_sent_prob(sentence, bigram=True, use_log=True, smoothing=True)
    print(f" Log Prob (bigram smoothed): {sentence_bigram_prob_smoothed:.6f}")

    sentence_bigram_prob_smoothed = ngram_model.calc_sent_prob(sentence, bigram=True, use_log=False, smoothing=True)
    print(f" Prob (bigram smoothed): {sentence_bigram_prob_smoothed:.6f}")
    print()

    sentence = "I don't have any idea of who kunwoo park is."
    print(f" Target: {sentence}")

    print()
    sentence_unigram_prob = ngram_model.calc_sent_prob(sentence, bigram=False, use_log=False)
    print(f" Prob (unigram): {sentence_unigram_prob:.6f}")

    sentence_unigram_logprob = ngram_model.calc_sent_prob(sentence, bigram=False, use_log=True)
    print(f" Log Prob (unigram): {sentence_unigram_logprob:.6f}")

    sentence_bigram_prob_unsmoothed = ngram_model.calc_sent_prob(sentence, bigram=True, use_log=True, smoothing=False)
    print(f" Log Prob (bigram unsmoothed): {sentence_bigram_prob_unsmoothed:.6f}")

    sentence_bigram_prob_smoothed = ngram_model.calc_sent_prob(sentence, bigram=True, use_log=True, smoothing=True)
    print(f" Log Prob (bigram smoothed): {sentence_bigram_prob_smoothed:.6f}")

    sentence_bigram_prob_smoothed = ngram_model.calc_sent_prob(sentence, bigram=True, use_log=False, smoothing=True)
    print(f" Prob (bigram smoothed): {sentence_bigram_prob_smoothed:.6f}")
    print()

    print(" 2-3. Generation")
    print()
    context = "NLP is so fun"
    for max_iter in range(10, 16, 5):
        generated_sentence = ngram_model.generate(context, max_iter, bigram=False)
        print(f' (context: "{context}", unigram, max_iter: {max_iter}): {generated_sentence}')
        generated_sentence = ngram_model.generate(context, max_iter, bigram=True)
        print(f' (context: "{context}", bigram, max_iter: {max_iter}): {generated_sentence}')

    print()
    context = "but I"
    for max_iter in range(10, 16, 5):
        generated_sentence = ngram_model.generate(context, max_iter, bigram=False)
        print(f' (context: "{context}", unigram, max_iter: {max_iter}): {generated_sentence}')
        generated_sentence = ngram_model.generate(context, max_iter, bigram=True)
        print(f' (context: "{context}", bigram, max_iter: {max_iter}): {generated_sentence}')
