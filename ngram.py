from collections import defaultdict, Counter

# Sample dataset
queries = [
    "میدان معلم",
    "میدان معلم شلوغ",
    "آخر امامزاده حسن",
    "شمس",
    "حرم حضرت م",
    "حر۱۳/۵۸",
    "سفارت ایتالیا",
    "امتم حسن",
    "افسریه جنوبی",
    "فردو",
    "میدان خرمشهر ۱۶متری دوم کوچه",
    # "میدان خرمشهر ۱۶متری دوم کوچه"
    "معلم خوب"
]


# # Assume queries are as defined previously
#
# def build_bigram_model(queries):
#     # Build a bi-gram model with query boundaries considered
#     bigram_model = defaultdict(Counter)
#     for query in queries:
#         tokens = ['<start>'] + query.split()
#         for i in range(len(tokens) - 1):
#             first, second = tokens[i], tokens[i + 1]
#             bigram_model[first][second] += 1
#     # Consider the end of query as well for predicting the start of new queries
#     bigram_model[tokens[-1]]['<end>'] += 1
#     return bigram_model
#
#
# def predict_next_word(bigram_model, input_text, top_n=4):
#     # Adjusted to handle last word of the input text for predictions
#     tokens = input_text.split()
#     last_word = tokens[-1] if tokens else '<start>'
#     suggested_words = bigram_model[last_word]
#     most_common = suggested_words.most_common(top_n)
#     # Filter out the <end> token in suggestions
#     return [word for word, count in most_common if word != '<end>']
#
#
# # Build the bi-gram model considering each query separately
# bigram_model = build_bigram_model(queries)
#
# # Example predictions
# input_text = "میدان معلم"
# suggestions = predict_next_word(bigram_model, input_text)
# print(f"Suggestions for '{input_text}': {suggestions}")


# Sample dataset as defined previously

def build_trigram_model(queries):
    # Build a tri-gram model with query boundaries considered
    trigram_model = defaultdict(Counter)
    for query in queries:
        # Add start tokens to handle the start of queries
        tokens = ['<start1>', '<start2>'] + query.split()
        for i in range(len(tokens)-2):
            first, second, third = tokens[i], tokens[i+1], tokens[i+2]
            trigram_model[(first, second)][third] += 1
    # Consider the end of query as well for predicting the start of new queries
    trigram_model[(tokens[-2], tokens[-1])]['<end>'] += 1
    return trigram_model

def predict_next_word(trigram_model, input_text, top_n=4):
    # Adjusted to handle the last two words of the input text for predictions
    tokens = ['<start1>', '<start2>'] + input_text.split()
    last_two_words = (tokens[-2], tokens[-1])
    suggested_words = trigram_model[last_two_words]
    most_common = suggested_words.most_common(top_n)
    # Filter out the <end> token in suggestions
    return [word for word, count in most_common if word != '<end>']

# Build the tri-gram model considering each query separately
trigram_model = build_trigram_model(queries)

# Example predictions
input_text = "معلم"
suggestions = predict_next_word(trigram_model, input_text)
print(f"Suggestions for '{input_text}': {suggestions}")
