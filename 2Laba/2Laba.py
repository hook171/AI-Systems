import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
import re
import string

# загрузка ресурсов nltk
nltk.data.path.append("C:/Users/ARTEM/AppData/Roaming/nltk_data") 
nltk.download('punkt')
nltk.download('stopwords')

text = """
Тестовая тестовая, тест запрос текст.
"""

# 1. Перевод текста в нижний регистр
text = text.lower()

# 2. Удаление знаков препинания
text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)

# 3. Удаление стоп-слов
stop_words = set(stopwords.words('russian'))
words = word_tokenize(text)
filtered_words = [word for word in words if word not in stop_words]

# 4. Удаление лишних символов (например, цифр)
filtered_words = [re.sub(r'\d+', '', word) for word in filtered_words]
filtered_words = [word for word in filtered_words if word.strip()]

# 5. Преобразование текста в мешок слов
from collections import Counter

bag_of_words = Counter(filtered_words)
print("Bag of Words:")
print(bag_of_words)

# 6. Преобразование текста в N-граммы, где N=2,4
# Биграммы (N=2)
bigrams = list(ngrams(filtered_words, 2))

# Четырехграммы (N=4)
fourgrams = list(ngrams(filtered_words, 4))

print("\nBigrams:")
print(bigrams)

print("\nFourgrams:")
print(fourgrams)