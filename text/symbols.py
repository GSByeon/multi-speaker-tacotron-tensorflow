'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run
through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details.
'''
from jamo import h2j, j2h
from jamo.jamo import _jamo_char_to_hcj

from .korean import ALL_SYMBOLS, PAD, EOS

#symbols = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!\'(),-.:;? '
symbols = ALL_SYMBOLS
