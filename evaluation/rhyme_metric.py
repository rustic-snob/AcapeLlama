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
        first_vowels_dict = {cumulative_length: char_to_replaced_vowel(word[0])}
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
        combined_rhymes[key] = list(combined_rhymes[key])
        
    combined_rhymes.pop('-', None)

    return combined_rhymes