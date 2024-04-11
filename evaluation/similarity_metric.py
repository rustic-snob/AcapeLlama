import torch
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from rouge import Rouge
import numpy as np

# semantic similarity
def eval_semantic_sim(model, tokenizer, golden_label, predicted_label, max_length = 512):
    golden_label_string = ' '.join(golden_label)
    predicted_label_string = ' '.join(predicted_label)
    
    print("golden_label_string:", golden_label_string)
    print("predicted_label_string:", predicted_label_string)
    
    # Encode the labels and convert to tensors
    golden_label_encoded = tokenizer.encode_plus(golden_label_string, add_special_tokens=True, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)
    predicted_label_encoded = tokenizer.encode_plus(predicted_label_string, add_special_tokens=True, return_tensors='pt', padding='max_length', truncation=True, max_length=max_length)

    # Get the embeddings
    with torch.no_grad():
        golden_label_output = model(**golden_label_encoded)
        predicted_label_output = model(**predicted_label_encoded)
        
    # Apply mean pooling to get a single vector per label
    golden_label_embeddings = golden_label_output[0].mean(dim=1)
    predicted_label_embeddings = predicted_label_output[0].mean(dim=1)

    # Calculate cosine similarity
    similarity = cosine_similarity(golden_label_embeddings, predicted_label_embeddings)

    # Print the similarity score
    return similarity

# lexical similarity
def get_full_lyrics_list(full_dataframe):
    # 'lyrics' 열의 값들을 리스트로 변환 후 set으로 중복 제거
    unique_lyrics_set = set(full_dataframe['lyrics'])
    # 중복이 제거된 가사들을 리스트로 변환
    lyrics_list = list(unique_lyrics_set)
    print("num of songs : ", len(lyrics_list))
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

def eval_lexical_sim(generated_line, original_lyrics, inverted_normalized_df_dict):
    original_lyrics = ' '.join(original_lyrics)
    generated_line = ' '.join(generated_line)
    original_words = original_lyrics.split()
    generated_words = generated_line.split()

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