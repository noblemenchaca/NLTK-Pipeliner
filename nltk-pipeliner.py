import nltk
from nltk.corpus import brown


#Part 1. NLTK Pipeliner
f = open('suess.txt')
raw = f.read()
print("Imported Text: " + raw + '\n')

text = nltk.word_tokenize(raw)
tagged = nltk.pos_tag(text)
print('POS Tagging: ' + str(tagged) + '\n')

lemmer = nltk.WordNetLemmatizer()
lemmas= [lemmer.lemmatize(t) for t in text]
print("Lemmas: " + str(lemmas))



#===================================================================
#Part 2.

browntags = brown.tagged_words(categories='news', tagset='universal')
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_sents = brown.sents(categories='news')

patterns = [
    (r'.*ing$', 'VBG'),               # gerunds
    (r'.*ed$', 'VBD'),                # simple past
    (r'.*es$', 'VBZ'),                # 3rd singular present
    (r'.*ould$', 'MD'),               # modals
    (r'.*\'s$', 'NN$'),               # possessive nouns
    (r'.*s$', 'NNS'),                 # plural nouns
    (r'^-?[0-9]+(.[0-9]+)?$', 'CD'),  # cardinal numbers    - 0.08006643196690336
    (r'.*s\'$', 'NNS'),               # 0.08006643196690336 - plural common genetive nouns
    (r'.*th$', 'OD'),                 # 0.08100125305805836 - ordinal number words
    (r'.*dy\'s$', 'PN$'),             # 0.08100125305805836 - possesive pronouns
    (r'.+self$', 'PPL'),              # 0.08135926964616028 - reflexive pronouns
    (r'.+days$', 'NRS'),              # 0.08135926964616028 - singular adverbial genetive nouns
    (r'.*tion$', 'NN'),               # 0.09050858245320922 - common noun
    (r'.*ly$', 'RB'),                 # 0.09763907949957237 - adverbs
    (r'.*gy$', 'NN'),                 # 0.09776836326749806 - more common nouns

    ]
#without the backoff     0.09776836326749806
#with the backoff tagger 0.2097380511963721
regexp_tagger = nltk.RegexpTagger(patterns, backoff=nltk.DefaultTagger('NN'))
example = regexp_tagger.tag(brown_sents[3])
print(example)
print(regexp_tagger.evaluate(brown_tagged_sents))


#========================================================================================
#part 3


def performance(cfd, wordlist):
    lt = dict((word, cfd[word].max()) for word in wordlist)
    baseline_tagger = nltk.UnigramTagger(model=lt, backoff=nltk.DefaultTagger('NN'))
    return baseline_tagger.evaluate(brown.tagged_sents(categories='news'))

def display():
    import pylab
    word_freqs = nltk.FreqDist(brown.words(categories='news')).most_common()
    words_by_freq = [w for (w, _) in word_freqs]
    cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
    sizes = 2 ** pylab.arange(15)
    perfs = [performance(cfd, words_by_freq[:size]) for size in sizes]
    pylab.plot(sizes, perfs, '-bo')
    pylab.title('Lookup Tagger Performance with Varying Model Size')
    pylab.xlabel('Model Size')
    pylab.ylabel('Performance')
    pylab.show()

#94.03% correctness at 16403  samples
display()

#========================================================================================
#Part 4 CFD and Dictionarires
cfd = nltk.ConditionalFreqDist(brown.tagged_words(categories='news'))
dict = {}
for key in cfd.keys():
    if len(cfd[key])  > 1:
        dict[key] = cfd.get(key)
#print(dict.keys())
print(repr(cfd.get('play')))

print(str(len(dict)/len(cfd)))
#.1197
