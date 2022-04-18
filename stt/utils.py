# import os
# from spellchecker import SpellChecker



# class SpellingCorrector:
#     def __init__(self, additional_vocab=''):
#         self.spell = SpellChecker()
#         if additional_vocab != '':
#             self.spell.word_frequency.load_text_file(additional_vocab)

#     def correct_folder(self, input_dir):
#         for file in os.listdir(input_dir):
#             with open(f'{input_dir}/{file}', 'r') as f:
#                 words = f.read().strip('\n').split(" ")
                
#             corrected = [self.spell.correction(word) for word in words]
#             new_words = " ".join(corrected)
#             with open(f'{input_dir}/{file}', 'w') as f:
#                 f.write(new_words)
    