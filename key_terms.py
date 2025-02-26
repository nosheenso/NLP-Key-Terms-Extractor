import nltk
from lxml import etree
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from string import punctuation

class KeyTermsExtractor:
    def __init__(self, xml_file):
        self.filter_items = set(punctuation) | set(stopwords.words('english'))
        self.xml_file = xml_file
        self.headers = []
        self.texts = []
        self.key_words = self.compute_tf_idf()

    def extract_data(self):
        """"Extracts headers and texts from the XML file."""
        root = etree.parse(self.xml_file).getroot()
        for item in root[0]:
            header, text = [item.find(f'value[@name="{tag}"]').text for tag in ("head", "text")]
            self.headers.append(header)
            tokens = self.process_text(text)
            self.texts.append(' '.join(tokens))

    def print_key_words(self):
        for header, key_words in self.key_words.items():
            print(f'{header}:')
            print(*[key_word[0] for key_word in key_words], '\n')

    def process_text(self, some_text: str) -> list:
        """Tokenizes, lemmatizes, and filters stop words and punctuation."""
        tokens = nltk.tokenize.word_tokenize(some_text.lower())
        lemmatizer = WordNetLemmatizer()
        filtered_tokens = list(filter(lambda x: x not in self.filter_items, tokens))
        processed_tokens = filter(lambda x: nltk.pos_tag([x])[0][1] == 'NN',
                                  [lemmatizer.lemmatize(token) for token in filtered_tokens])
        return list(processed_tokens)

    def compute_tf_idf(self) -> dict:
        """Computes TF-IDF and returns top 5 important terms per text."""
        self.extract_data()
        tfidf_scores = dict()
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.texts)
        terms = vectorizer.get_feature_names_out()
        for i in range(len(self.texts)):
            words_score = list(((term, score) for term, score in zip(terms, tfidf_matrix.toarray()[i])))
            top_five = sorted(words_score, key=lambda x: (x[1], x[0]), reverse=True)[:5]
            tfidf_scores[self.headers[i]] = top_five
        return tfidf_scores


def main():
    filename = 'news.xml'
    new = KeyTermsExtractor(filename)
    new.print_key_words()


if __name__ == '__main__':
    main()