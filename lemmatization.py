from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
import re
import nltk
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')

def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemma(filepath_in, filepath_out):
    fin = open(filepath_in, "r", encoding='UTF-8')
    fout = open(filepath_out, "w", encoding='UTF-8')
    cnt = 0
    for line in fin:
        cnt += 1
        sentence = re.sub("[#]+", "#", line)
        sentence = re.sub("#.#", "#", sentence)
        tokens = word_tokenize(sentence)
        tagged_sent = pos_tag(tokens)

        wnl = WordNetLemmatizer()
        lemmas_sent = []
        for tag in tagged_sent:
            wordnet_pos = get_wordnet_pos(tag[1]) or wordnet.NOUN
            lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))

        fout.write(" ".join(lemmas_sent))
        fout.write('\n')

if __name__ == "__main__":
    filepath1 = "C:/Users/Eleanor/Desktop/textsum-transformer-master/sumdata/train/train.title_01"
    filepath2 = "C:/Users/Eleanor/Desktop/textsum-transformer-master/sumdata/train/train.title_01_new.txt"
    lemma(filepath1, filepath2)
