import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
import numpy as np

# semantic similarity
def eval_semantic_sim(model, tokenizer, golden_lyrics, predict_mungchi_string, max_length = 512):
    golden_lyrics_string = golden_lyrics
    predict_mungchi_string = predict_mungchi_string.replace(' / ', ' ')
    
    # Encode the labels and convert to tensors
    golden_lyrics_encoded = tokenizer.encode_plus(golden_lyrics_string, add_special_tokens=True, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
    predict_mungchi_string_encoded = tokenizer.encode_plus(predict_mungchi_string, add_special_tokens=True, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
    # golden_lyrics_encoded = tokenizer.encode_plus(golden_lyrics_string, add_special_tokens=True, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
    # predict_mungchi_string_encoded = tokenizer.encode_plus(predict_mungchi_string, add_special_tokens=True, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)

    # Get the embeddings
    with torch.no_grad():
        golden_lyrics_output = model(**golden_lyrics_encoded)
        predict_mungchi_string_output = model(**predict_mungchi_string_encoded)
        
    # Apply mean pooling to get a single vector per label
    golden_lyrics_embeddings = golden_lyrics_output[0][:, 0, :]
    predict_mungchi_string_embeddings = predict_mungchi_string_output[0][:, 0, :]

    # Calculate cosine similarity
    similarity = cosine_similarity(golden_lyrics_embeddings, predict_mungchi_string_embeddings)

    # Print the similarity score
    return similarity[0][0]

# lexical similarity
def eval_lexical_sim_levenshtein(golden_lyrics, predict_mungchi_string):
    golden_lyrics_string = golden_lyrics
    predict_mungchi_string = predict_mungchi_string.replace(' / ', ' ')
    
    # Calculate the Levenshtein distance
    return levenshtein_distance(golden_lyrics_string, predict_mungchi_string)

def eval_our_lexical_sim(golden_lyrics, predict_mungchi_string):
    golden_lyrics_string = golden_lyrics.replace('\n\n', '\n')
    golden_lyrics_string = golden_lyrics_string.replace('\n', ' ')
    predict_mungchi_string = predict_mungchi_string.replace(' / ', ' ')
    
    golden_lyrics_list = golden_lyrics_string.split(' ')
    predict_mungchi_list = predict_mungchi_string.split(' ')
    
    similarity_matrix = get_similarity_matrix(golden_lyrics_list, predict_mungchi_list)
    
    final_similarity_score, score_list = calculate_similarity_score(similarity_matrix)
    
    return final_similarity_score, score_list

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def levenshtein_similarity(s1, s2):
    distance = levenshtein_distance(s1, s2)
    # Normalize the distance to get a similarity score between 0 and 1
    # The similarity is higher when the Levenshtein distance is smaller.
    # If the strings are identical, the distance is 0, so the similarity is 1.
    similarity = 1 - distance / max(len(s1), len(s2))
    return similarity

def get_similarity_matrix(original_lyrics, generated_lyrics):
# Placeholder lists for demonstration, these should be replaced with actual data extracted from the images

    # Initialize an empty matrix with dimensions based on the lists' lengths.
    similarity_matrix = np.zeros((len(original_lyrics), len(generated_lyrics)))

    # Fill the similarity matrix with the Levenshtein similarity scores.
    for i, original in enumerate(original_lyrics):
        for j, generated in enumerate(generated_lyrics):
            similarity_matrix[i][j] = levenshtein_similarity(original, generated)

    # Display the similarity matrix
    return similarity_matrix

def calculate_similarity_score(similarity_matrix):
    score_list = []

    while similarity_matrix.size > 0:
        # Find the indices of the maximum value in the similarity matrix.
        i, j = np.unravel_index(similarity_matrix.argmax(), similarity_matrix.shape)
        # Append the highest similarity score to the score list.
        score_list.append(similarity_matrix[i][j])
        # Delete the corresponding row and column.
        similarity_matrix = np.delete(similarity_matrix, i, 0)  # Delete row
        similarity_matrix = np.delete(similarity_matrix, j, 1)  # Delete column

    # Calculate the average of the score list as the final similarity score.
    final_similarity_score = np.mean(score_list)
    return final_similarity_score, score_list

def eval_lexical_sim_precision(golden_lyrics, predict_mungchi_string):
    # 문장을 띄어쓰기 기준으로 토큰화
    predict_mungchi = predict_mungchi_string.replace(' / ', ' ')
    
    long_tokens = set(golden_lyrics.split())
    short_tokens = predict_mungchi.split()
    
    # 짧은 문장의 토큰이 긴 문장의 토큰 목록에 몇 번 등장하는지 세기
    common_tokens_count = sum(1 for token in short_tokens if token in long_tokens)
    
    # 정밀도 계산: 긴 문장에 포함된 짧은 문장 토큰의 수를 짧은 문장의 토큰 수로 나눔
    if len(short_tokens) == 0:
        return 0  # 짧은 문장에 토큰이 없으면 정밀도는 0
    precision = common_tokens_count / len(short_tokens)
    return precision

def calculate_bleu(reference_texts, candidate_text):
    reference_tokens = [ref.split() for ref in reference_texts]
    candidate_tokens = candidate_text.split()
    return sentence_bleu(reference_tokens, candidate_tokens)

def eval_lexical_sim_bleu(golden_lyrics, predict_mungchi_string):
    golden_lyrics_string_in_list = [golden_lyrics]
    predict_mungchi_string = predict_mungchi_string.replace(' / ', ' ')
    
    return calculate_bleu(golden_lyrics_string_in_list, predict_mungchi_string)

def cosine_sim(text1, text2):
    vectorizer = CountVectorizer()
    tfidf = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf)[0, 1]

def eval_lexical_sim_cos(golden_lyrics, predict_mungchi_string):
    golden_lyrics_string = golden_lyrics
    predict_mungchi_string = predict_mungchi_string.replace(' / ', ' ')
    
    return cosine_sim(golden_lyrics_string, predict_mungchi_string)

def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

def eval_lexical_sim_jaccard(golden_lyrics, predict_mungchi_string):
    golden_lyrics_string = golden_lyrics
    predict_mungchi_string = predict_mungchi_string.replace(' / ', ' ')
    
    set1 = set(golden_lyrics_string.split())
    set2 = set(predict_mungchi_string.split())
    
    return jaccard_similarity(set1, set2)

def calculate_rouge(hypothesis, reference):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference)
    return scores

def eval_lexical_sim_rouge(golden_lyrics, predict_mungchi_string):
    golden_lyrics_string = golden_lyrics
    predict_mungchi_string = predict_mungchi_string.replace(' / ', ' ')
    rouge_scores = calculate_rouge(golden_lyrics_string, predict_mungchi_string)
    rouge_unigram_recall = rouge_scores[0]['rouge-1']['r']
    rouge_bigram_recall = rouge_scores[0]['rouge-2']['r']
    
    return rouge_unigram_recall, rouge_bigram_recall

def print_lexical_sim_total(golden_lyrics, predict_mungchi_string):
    print(f"rouge : {eval_lexical_sim_rouge(golden_lyrics, predict_mungchi_string)}")
    print(f"levenshtein: {eval_lexical_sim_levenshtein(golden_lyrics, predict_mungchi_string)}")
    print(f"cosine: {eval_lexical_sim_cos(golden_lyrics, predict_mungchi_string)}")
    print(f"jaccard: {eval_lexical_sim_jaccard(golden_lyrics, predict_mungchi_string)}")

# with tf-idf matrix similarity
def get_full_lyrics_list(full_dataframe):
    # 'lyrics' 열의 값들을 리스트로 변환 후 set으로 중복 제거
    unique_lyrics_set = set(full_dataframe['lyrics'])
    # 중복이 제거된 가사들을 리스트로 변환
    lyrics_list = list(unique_lyrics_set)
    return lyrics_list

def get_inverted_tfidf_dictionary(corpus):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(corpus)
    
    tfidf_df = (vectorizer.transform(corpus) > 0).sum(axis=0).tolist()[0]
    inverted_normalized_df = [1 - (value/len(corpus))*2 for value in tfidf_df]
    inverted_normalized_df_dict = dict(zip(vectorizer.get_feature_names_out(), inverted_normalized_df))
    return inverted_normalized_df_dict

def calculate_triplet(word_gen, word_orig, inverted_normalized_df_dict):
    """
    Calculate the triplet (ROUGE score, tf-idf score, checked status) for a given pair of words.

    Args:
    - word_gen (str): The word from the generated line.
    - word_orig (str): The word from the original lyrics.
    - inverted_normalized_df_dict (dict): A dictionary containing the tf-idf scores for each word in the original lyrics.

    Returns:
    - tuple: A triplet containing the ROUGE score, tf-idf score, and checked status.
    """
    word_gen_spaced = ' '.join(word_gen)
    word_orig_spaced = ' '.join(word_orig)

    # Calculate ROUGE score
    rouge = Rouge()
    scores = rouge.get_scores(word_gen_spaced, word_orig_spaced)
    rouge_score = scores[0]['rouge-1']['f']

    # Get tf-idf score
    tfidf_score = inverted_normalized_df_dict.get(word_orig, 0.0)

    # Set checked status (initially set to False)
    checked = False

    return (rouge_score, tfidf_score, checked)

def eval_lexical_sim(generated_line, original_lyrics_string, inverted_normalized_df_dict):
    generated_line = ' '.join(generated_line)
    generated_words = generated_line.split()
    
    lyrics_string_only_space = original_lyrics_string.replace('\n', ' ')
    original_words = lyrics_string_only_space.split()


    # Construct the score matrix using list comprehensions
    scores_matrix = np.array([[calculate_triplet(w_g, w_o, inverted_normalized_df_dict) for w_o in original_words] for w_g in generated_words])

    # Determine penalty_diverse
    # max_indices = np.argmax(scores_matrix[:,:,0], axis=1)
    # penalty_diverse = len(max_indices) - len(np.unique(max_indices))

    scores = []

    for _ in range(len(generated_words)):
        # Find the highest score in the matrix
        i, j = np.unravel_index(np.argmax(scores_matrix[:,:,0], axis=None), scores_matrix[:,:,0].shape)
        max_score = scores_matrix[i, j, 0]
        scores.append(max_score * scores_matrix[i, j, 1])

        # Set the entire corresponding row and column to -1
        scores_matrix[i, :, 0] = -1
        scores_matrix[:, j, 0] = -1

    # Calculate the final score
    final_score = (sum(scores) / len(scores)) * ((len(generated_words)) / len(generated_words))

    return final_score