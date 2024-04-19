import pandas as pd
from hangul_utils import split_syllables

# words -> first/last/side vowel 변환
def char_to_replaced_vowel(char):
    vowel_mappings = {'ㅏ': 'ㅏ', 'ㅑ': 'ㅑ', 'ㅓ': 'ㅓ', 'ㅕ': 'ㅕ', 'ㅗ': 'ㅗ', 'ㅛ': 'ㅛ', 
                        'ㅜ': 'ㅜ', 'ㅠ': 'ㅠ', 'ㅡ': 'ㅡ', 'ㅣ': 'ㅣ',
                        'ㅐ': 'ㅐ', 'ㅒ': 'ㅐ', 'ㅔ': 'ㅐ', 'ㅖ': 'ㅐ', 
                        'ㅘ': 'ㅏ', 'ㅙ': 'ㅐ', 'ㅚ': 'ㅐ', 
                        'ㅝ': 'ㅓ', 'ㅞ': 'ㅐ', 'ㅟ': 'ㅣ', 'ㅢ': 'ㅣ'}
    try:
        return vowel_mappings[split_syllables(char)[1]]
    except:
        return '-'
    
def get_first_vowels_dicts(words):
    '''
    input : words
    words -> first_chars -> first_vowels
    output : first_vowels_dicts
    '''
    first_vowels_dicts = []
    cumulative_length = 0

    for word in words:
        try:
            first_vowels_dict = {cumulative_length: char_to_replaced_vowel(word[0])}
        except:
            print("words : ", words)
            print("word : ", word)
            break
        first_vowels_dicts.append(first_vowels_dict)
        cumulative_length += len(word)
        
    return first_vowels_dicts

def get_last_vowels_dicts(words):
    '''
    input : words
    words -> last_chars -> last_vowels
    output : last_vowels_dicts
    '''
    last_vowels_dicts = []

    # Initialize a variable to track the cumulative length of the words processed, starting from 0
    cumulative_length = 0

    for word in words:
        # The index for the last character of a word is cumulative_length + len(word) - 1
        last_vowels_index = cumulative_length + len(word) - 1
        last_vowels_dict = {last_vowels_index: char_to_replaced_vowel(word[-1])}  # Create a dictionary with this index as key and the last character as value
        last_vowels_dicts.append(last_vowels_dict)
        
        # Update the cumulative length with the length of the current word for the next iteration
        cumulative_length += len(word)

    return last_vowels_dicts

def get_side_vowels_dicts(words):
    side_vowels_dicts = []

    # Re-initialize the variable to track the cumulative length of the words processed, starting from 0
    cumulative_length = 0

    for word in words:
        # The index for the first character of a word
        first_vowels_index = cumulative_length
        # The index for the last character of a word is cumulative_length + len(word) - 1
        last_vowels_index = cumulative_length + len(word) - 1
        
        # Create a dictionary with both first and last character indexes as keys
        # and the corresponding characters as values
        vowels_dict = {first_vowels_index: char_to_replaced_vowel(word[0]), last_vowels_index: char_to_replaced_vowel(word[-1])}
        side_vowels_dicts.append(vowels_dict)
        
        # Update the cumulative length with the length of the current word for the next iteration
        cumulative_length += len(word)

    return side_vowels_dicts

# Find first/last/side Rhyme
def get_vowel_groups_with_duplicates(vowel_dicts):
    vowel_groups = {}

    # Iterating through each dictionary in the list
    for vowel_dict in vowel_dicts:
        # Extracting the key-value pair from each dictionary
        for key, value in vowel_dict.items():
            # Grouping keys by the same vowel value
            if value not in vowel_groups:
                vowel_groups[value] = []
            vowel_groups[value].append(key)

    # Filtering out the entries that do not have duplicates
    vowel_groups_with_duplicates = {k: v for k, v in vowel_groups.items() if len(v) > 1}

    return vowel_groups_with_duplicates

def get_vowel_groups_within_duplicates(vowels_dicts):
    # To find and group keys by the same character value when the key difference is more than 1
    vowel_groups = {}

    # Iterating through each dictionary in the list
    for vowel_dict in vowels_dicts:
        keys = list(vowel_dict.keys())
        # Check if the difference between the keys is more than 1 and if the values are the same
        if len(keys) > 1 and abs(keys[0] - keys[1]) > 1 and vowel_dict[keys[0]] == vowel_dict[keys[1]]:
            value = vowel_dict[keys[0]]
            if value not in vowel_groups:
                vowel_groups[value] = []
            vowel_groups[value].extend(keys)

    return vowel_groups

# Combine Rhymes
def combine_rhyme_dicts(first_vowels_rhyme, last_vowels_rhyme, side_vowels_rhyme):
    combined_rhymes = {}

    # Merging the dictionaries
    for d in [first_vowels_rhyme, last_vowels_rhyme, side_vowels_rhyme]:
        for key, value in d.items():
            if key not in combined_rhymes:
                combined_rhymes[key] = set()
            combined_rhymes[key].update(value)

    # Convert sets back to lists
    for key in combined_rhymes:
        combined_rhymes[key] = sorted(list(combined_rhymes[key]))
        
    combined_rhymes.pop('-', None)

    return combined_rhymes

# rhyme은 L줄 단위
def get_dataframe_sample_only_L(dataframe):
    df_sampling_L = dataframe.loc[(dataframe['sampling_strategy'] == 'L')].reset_index(drop=True)
    return df_sampling_L

def get_rhyme_column(df_sampling_L):
    rhymes_list_by_song_TOTAL_SONG = []

    # 곡 단위
    for i in range(df_sampling_L.shape[0]):

        rhymes_list_by_sample_ONE_SONG = []

        ONE_SONG = df_sampling_L.iloc[i][:]
        ONE_LABEL = ONE_SONG['label']

        # 샘플 단위
        for j in range(len(ONE_LABEL)):
            ONE_SAMPLE = ONE_LABEL[j]
            first_vowels_dicts = get_first_vowels_dicts(ONE_SAMPLE)
            last_vowels_dicts  =  get_last_vowels_dicts(ONE_SAMPLE)
            side_vowels_dicts  =  get_side_vowels_dicts(ONE_SAMPLE)
            
            first_vowels_rhyme =   get_vowel_groups_with_duplicates(first_vowels_dicts)
            last_vowels_rhyme  =   get_vowel_groups_with_duplicates(last_vowels_dicts )
            side_vowels_rhyme  = get_vowel_groups_within_duplicates(side_vowels_dicts )
            
            total_rhymes_ONE_SAMPLE = combine_rhyme_dicts(first_vowels_rhyme, last_vowels_rhyme, side_vowels_rhyme)
            rhymes_list_by_sample_ONE_SONG.append(total_rhymes_ONE_SAMPLE)
            
        rhymes_list_by_song_TOTAL_SONG.append(rhymes_list_by_sample_ONE_SONG)
        
    rhyme_column = pd.DataFrame({'rhyme': rhymes_list_by_song_TOTAL_SONG})
    return rhyme_column 

def get_rhyme_in_lyrics(title_of_song, s_th_sample, rhyme_df):
    # 한 줄 뭉치에 대한 행
    sample_info_line_mungchi = rhyme_df.loc[(rhyme_df['title']==title_of_song) & (rhyme_df['chunk_strategy'] == 'line')]
    sample_label = sample_info_line_mungchi.iloc[0]['label'][s_th_sample]
    
    # 단어 뭉치에 대한 행
    sample_info_word_mungchi = rhyme_df.loc[(rhyme_df['title']==title_of_song) & (rhyme_df['chunk_strategy'] == 'word')]
    sample_rhyme = sample_info_word_mungchi.iloc[0]['rhyme'][s_th_sample]


    # 한 줄 뭉치 기준, 뭉치 당 개행으로 나눈 가사 string
    sample_lyrics_string = '\n'.join(sample_label)
    # 모든 글자를 원소로 하는 list 생성
    sample_char_list = list(sample_lyrics_string)
    print(f"[original_lyrics]\n{sample_lyrics_string}")

    # 공백과 개행 문자를 제거하면서 위치를 기록
    sample_char_list_only_word = []
    space_newline_positions = []
    for index, char in enumerate(sample_char_list):
        if char in [' ', '\n']:
            space_newline_positions.append((index, char))
        else:
            sample_char_list_only_word.append(char)

    # 딕셔너리의 키(대체할 문자)와 값(인덱스 목록)에 따라 문자열의 특정 위치를 변경
    for key, indices in sample_rhyme.items():
        for index in indices:
            sample_char_list_only_word[index] = key

    # 공백과 개행을 원래 위치에 다시 추가
    reconstructed_list = sample_char_list_only_word[:]  # 복사본 생성
    for position, char in space_newline_positions:
        reconstructed_list.insert(position, char)  # 원래 위치에 직접 삽입
        
    # 결과를 문자열로 변환
    reconstructed_string = ''.join(reconstructed_list)
    print(f"\n\n[rhyme_in_lyrics]\n{reconstructed_string}")
    