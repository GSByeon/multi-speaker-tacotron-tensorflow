import os
import string
import argparse
import operator
from functools import partial
from difflib import SequenceMatcher

from audio.get_duration import get_durations
from text import remove_puncuations, text_to_sequence
from utils import load_json, write_json, parallel_run, remove_postfix, backup_file

def plain_text(text):
    return "".join(remove_puncuations(text.strip()).split())

def add_punctuation(text):
    if text.endswith('다'):
        return text + "."
    else:
        return text

def similarity(text_a, text_b):
    text_a = plain_text(text_a)
    text_b = plain_text(text_b)

    score = SequenceMatcher(None, text_a, text_b).ratio()
    return score

def first_word_combined_words(text):
    words = text.split()
    if len(words) > 1:
        first_words = [words[0], words[0]+words[1]]
    else:
        first_words = [words[0]]
    return first_words

def first_word_combined_texts(text):
    words = text.split()
    if len(words) > 1:
        if len(words) > 2:
            text2 = " ".join([words[0]+words[1]] + words[2:])
        else:
            text2 = words[0]+words[1]
        texts = [text, text2]
    else:
        texts = [text]
    return texts

def search_optimal(found_text, recognition_text):
    # 1. found_text is usually more accurate
    # 2. recognition_text can have more or less word

    optimal = None

    if plain_text(recognition_text) in plain_text(found_text):
        optimal = recognition_text
    else:
        found = False

        for tmp_text in first_word_combined_texts(found_text):
            for recognition_first_word in first_word_combined_words(recognition_text):
                if recognition_first_word in tmp_text:
                    start_idx = tmp_text.find(recognition_first_word)

                    if tmp_text != found_text:
                        found_text = found_text[max(0, start_idx-1):].strip()
                    else:
                        found_text = found_text[start_idx:].strip()
                    found = True
                    break

            if found:
                break

        recognition_last_word = recognition_text.split()[-1]
        if recognition_last_word in found_text:
            end_idx = found_text.find(recognition_last_word)

            punctuation = ""
            if len(found_text) > end_idx + len(recognition_last_word):
                punctuation = found_text[end_idx + len(recognition_last_word)]
                if punctuation not in string.punctuation:
                    punctuation = ""

            found_text = found_text[:end_idx] + recognition_last_word + punctuation
            found = True

        if found:
            optimal = found_text

    return optimal


def align_text_fn(
        item, score_threshold, debug=False):

    audio_path, recognition_text = item

    audio_dir = os.path.dirname(audio_path)
    base_dir = os.path.dirname(audio_dir)

    news_path = remove_postfix(audio_path.replace("audio", "assets"))
    news_path = os.path.splitext(news_path)[0] + ".txt"

    strip_fn = lambda line: line.strip().replace('"', '').replace("'", "")
    candidates = [strip_fn(line) for line in open(news_path).readlines()]

    scores = { candidate: similarity(candidate, recognition_text) \
                    for candidate in candidates}
    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1))[::-1]

    first, second = sorted_scores[0], sorted_scores[1]

    if first[1] > second[1] and first[1] >= score_threshold:
        found_text, score = first
        aligned_text = search_optimal(found_text, recognition_text)

        if debug:
            print("   ", audio_path)
            print("   ", recognition_text)
            print("=> ", found_text)
            print("==>", aligned_text)
            print("="*30)

        if aligned_text is not None:
            result = { audio_path: add_punctuation(aligned_text) }
        elif abs(len(text_to_sequence(found_text)) - len(text_to_sequence(recognition_text))) > 10:
            result = {}
        else:
            result = { audio_path: [add_punctuation(found_text), recognition_text] }
    else:
        result = {}

    if len(result) == 0:
        result = { audio_path: [recognition_text] }

    return result

def align_text_batch(config):
    align_text = partial(align_text_fn,
            score_threshold=config.score_threshold)

    results = {}
    data = load_json(config.recognition_path)

    items = parallel_run(
            align_text, data.items(),
            desc="align_text_batch", parallel=True)

    for item in items:
        results.update(item)

    found_count = sum([type(value) == str for value in results.values()])
    print(" [*] # found: {:.5f}% ({}/{})".format(
            len(results)/len(data), len(results), len(data)))
    print(" [*] # exact match: {:.5f}% ({}/{})".format(
            found_count/len(items), found_count, len(items)))

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recognition_path', required=True)
    parser.add_argument('--alignment_filename', default="alignment.json")
    parser.add_argument('--score_threshold', default=0.4, type=float)
    config, unparsed = parser.parse_known_args()

    results = align_text_batch(config)

    base_dir = os.path.dirname(config.recognition_path)
    alignment_path = \
            os.path.join(base_dir, config.alignment_filename)

    if os.path.exists(alignment_path):
        backup_file(alignment_path)

    write_json(alignment_path, results)
    duration = get_durations(results.keys(), print_detail=False)
