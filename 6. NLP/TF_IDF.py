from sklearn.feature_extraction.text import TfidfVectorizer

# 샘플 문서 리스트
documents = ['The cat chased the mouse under the table.', 'The mouse found a piece of cheese.','The dog barked at the cat loudly.'
]

# TF-IDF 벡터라이저 초기화
vectorizer = TfidfVectorizer()

# 문서에 TF-IDF 적용
tfidf_matrix = vectorizer.fit_transform(documents)

# 단어 리스트 출력
print("단어 리스트:", vectorizer.get_feature_names_out())

# TF-IDF 행렬 출력
print("TF-IDF 행렬:\n", tfidf_matrix.toarray())
