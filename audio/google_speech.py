import io
import os
import sys
import json
import string
import argparse
import operator
import numpy as np
from glob import glob
from tqdm import tqdm
from nltk import ngrams
from difflib import SequenceMatcher
from collections import defaultdict

from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types

from utils import parallel_run
from text import text_to_sequence

####################################################
# When one or two audio is missed in the middle
####################################################

def get_continuous_audio_paths(paths, debug=False):
    audio_ids = get_audio_ids_from_paths(paths)
    min_id, max_id = min(audio_ids), max(audio_ids)

    if int(max_id) - int(min_id) + 1 != len(audio_ids):
        base_path = paths[0].replace(min_id, "{:0" + str(len(max_id)) + "d}")
        new_paths = [
                base_path.format(audio_id) \
                        for audio_id in range(int(min_id), int(max_id) + 1)]

        if debug: print("Missing audio : {} -> {}".format(paths, new_paths))
        return new_paths
    else:
        return paths

def get_argmax_key(info, with_value=False):
    max_key = max(info.keys(), key=(lambda k: info[k]))

    if with_value:
        return max_key, info[max_key]
    else:
        return max_key

def similarity(text_a, text_b):
    text_a = "".join(remove_puncuations(text_a.strip()).split())
    text_b = "".join(remove_puncuations(text_b.strip()).split())

    score = SequenceMatcher(None, text_a, text_b).ratio()
    #score = 1 / (distance(decompose_ko_text(text_a), decompose_ko_text(text_b)) + 1e-5)
    #score = SequenceMatcher(None, 
    #        decompose_ko_text(text_a), decompose_ko_text(text_b)).ratio()

    if len(text_a) < len(text_b):
        return -1 + score
    else:
        return score

def get_key_value_sorted(data):
    keys = list(data.keys())
    keys.sort()
    values = [data[key] for key in keys]
    return keys, values

def replace_pred_with_book(
        path, book_path=None, threshold=0.9, max_candidate_num=5,
        min_post_char_check=2, max_post_char_check=7, max_n=5,
        max_allow_missing_when_matching=4, debug=False):

    #######################################
    # find text book from pred
    #######################################

    if book_path is None:
        book_path = path.replace("speech", "text").replace("json", "txt")

    data = json.loads(open(path).read())

    keys, preds = get_key_value_sorted(data)

    book_words = [word for word in open(book_path).read().split() if word != "=="]
    book_texts = [text.replace('\n', '') for text in open(book_path).readlines()]

    loc = 0
    prev_key = None
    force_stop = False
    prev_end_loc = -1
    prev_sentence_ended = True

    prev_empty_skip = False
    prev_not_found_skip = False

    black_lists = ["160.{:04d}".format(audio_id) for audio_id in range(20, 36)]

    new_preds = {}
    for key, pred in zip(keys, preds):
        if debug: print(key, pred)

        if pred == "" or key in black_lists:
            prev_empty_skip = True
            continue

        width, counter = 1, 0
        sim_dict, loc_dict = {}, {}

        while True:
            words = book_words[loc:loc + width]

            if len(words) == 0:
                print("Force stop. Left {}, Del {} {}". \
                        format(len(preds) - len(new_preds), new_preds[prev_key], prev_key))
                new_preds.pop(prev_key, None)
                force_stop = True
                break

            candidate_candidates = {}

            for _pred in list(set([pred, koreanize_numbers(pred)])):
                max_skip = 0 if has_number(_pred[0]) or \
                        _pred[0] in """"'“”’‘’""" else len(words)

                end_sims = []
                for idx in range(min(max_skip, 10)):
                    text = " ".join(words[idx:])

                    ################################################
                    # Score of trailing sentence is also important
                    ################################################

                    for jdx in range(min_post_char_check,
                                     max_post_char_check):
                        sim = similarity(
                                "".join(_pred.split())[-jdx:],
                                "".join(text.split())[-jdx:])
                        end_sims.append(sim)

                    candidate_candidates[text] = similarity(_pred, text)

            candidate, sim = get_argmax_key(
                    candidate_candidates, with_value=True)

            if sim > threshold or max(end_sims + [-1]) > threshold - 0.2 or \
                    len(sim_dict) > 0:
                sim_dict[candidate] = sim
                loc_dict[candidate] = loc + width

            if len(sim_dict) > 0:
                counter += 1

            if counter > max_candidate_num:
                break

            width += 1

            if width - len(_pred.split()) > 5:
                break

        if force_stop:
            break

        if len(sim_dict) != 0:
            #############################################################
            # Check missing words between prev pred and current pred
            #############################################################

            if prev_key is not None:
                cur_idx = int(key.rsplit('.', 2)[-2])
                prev_idx = int(prev_key.rsplit('.', 2)[-2])

                if cur_idx - prev_idx > 10:
                    force_stop = True
                    break

            # word alinged based on prediction but may contain missing words
            # because google speech recognition sometimes skip one or two word
            # ex. ('오누이는 서로 자기가 할 일을 정했다.', '서로 자기가 할 일을 정했다.')
            original_candidate = new_candidate = get_argmax_key(sim_dict)

            word_to_find = original_candidate.split()[0]

            if not prev_empty_skip:
                search_idx = book_words[prev_end_loc:].index(word_to_find) \
                        if word_to_find in book_words[prev_end_loc:] else -1

                if 0 < search_idx < 4 and not prev_sentence_ended:
                    words_to_check = book_words[prev_end_loc:prev_end_loc + search_idx]

                    if ends_with_punctuation(words_to_check[0]) == True:
                        tmp = " ".join([new_preds[prev_key]] + words_to_check[:1])
                        if debug: print(prev_key, tmp, new_preds[prev_key])
                        new_preds[prev_key] = tmp

                        prev_end_loc += 1
                        prev_sentence_ended = True

                search_idx = book_words[prev_end_loc:].index(word_to_find) \
                        if word_to_find in book_words[prev_end_loc:] else -1

                if 0 < search_idx < 4 and prev_sentence_ended:
                    words_to_check = book_words[prev_end_loc:prev_end_loc + search_idx]

                    if not any(ends_with_punctuation(word) for word in words_to_check):
                        new_candidate = " ".join(words_to_check + [original_candidate])
                        if debug: print(key, new_candidate, original_candidate)

            new_preds[key] = new_candidate
            prev_sentence_ended = ends_with_punctuation(new_candidate)

            loc = loc_dict[original_candidate]
            prev_key = key
            prev_not_found_skip = False
        else:
            loc += len(_pred.split()) - 1
            prev_sentence_ended = True
            prev_not_found_skip = True

        prev_end_loc = loc
        prev_empty_skip = False

        if debug:
            print("=", pred)
            print("=", new_preds[key], loc)

    if force_stop:
        print(" [!] Force stop: {}".format(path))

    align_diff = loc - len(book_words)

    if abs(align_diff) > 10:
        print("   => Align result of {}: {} - {} = {}".format(path, loc, len(book_words), align_diff))

    #######################################
    # find exact match of n-gram of pred
    #######################################

    finished_ids = []

    keys, preds = get_key_value_sorted(new_preds)

    if abs(align_diff) > 10:
        keys, preds = keys[:-30], preds[:-30]

    unfinished_ids = range(len(keys))
    text_matches = []

    for n in range(max_n, 1, -1):
        ngram_preds = ngrams(preds, n)

        for n_allow_missing in range(0, max_allow_missing_when_matching + 1):
            unfinished_ids = list(set(unfinished_ids) - set(finished_ids))

            existing_ngram_preds = []

            for ngram in ngram_preds:
                for text in book_texts:
                    candidates = [
                            " ".join(text.split()[:-n_allow_missing]),
                            " ".join(text.split()[n_allow_missing:]),
                    ]
                    for tmp_text in candidates:
                        if " ".join(ngram) == tmp_text:
                            existing_ngram_preds.append(ngram)
                            break

            tmp_keys = []
            cur_ngram = []

            ngram_idx = 0
            ngram_found = False

            for id_idx in unfinished_ids:
                key, pred = keys[id_idx], preds[id_idx]

                if ngram_idx >= len(existing_ngram_preds):
                    break

                cur_ngram = existing_ngram_preds[ngram_idx]

                if pred in cur_ngram:
                    ngram_found = True

                    tmp_keys.append(key)
                    finished_ids.append(id_idx)

                    if len(tmp_keys) == len(cur_ngram):
                        if debug: print(n_allow_missing, tmp_keys, cur_ngram)

                        tmp_keys = get_continuous_audio_paths(tmp_keys, debug)
                        text_matches.append(
                                [[" ".join(cur_ngram)], tmp_keys]
                        )

                        ngram_idx += 1
                        tmp_keys = []
                        cur_ngram = []
                    else:
                        if pred == cur_ngram[-1]:
                            ngram_idx += 1
                            tmp_keys = []
                            cur_ngram = []
                else:
                    if len(tmp_keys) > 0:
                        ngram_found = False

                        tmp_keys = []
                        cur_ngram = []

    for id_idx in range(len(keys)):
        if id_idx not in finished_ids:
            key, pred = keys[id_idx], preds[id_idx]

            text_matches.append(
                    [[pred], [key]]
            )

    ##############################################################
    # ngram again for just in case after adding missing words
    ##############################################################

    max_keys = [max(get_audio_ids_from_paths(item[1], as_int=True)) for item in text_matches]
    sorted_text_matches = \
            [item for _, item in sorted(zip(max_keys, text_matches))]

    preds = [item[0][0] for item in sorted_text_matches]
    keys = [item[1] for item in sorted_text_matches]

    def book_sentence_idx_search(query, book_texts):
        for idx, text in enumerate(book_texts):
            if query in text:
                return idx, text
        return False, False

    text_matches = []
    idx, book_cursor_idx = 0, 0

    if len(preds) == 0:
        return []

    while True:
        tmp_texts = book_texts[book_cursor_idx:]

        jdx = 0
        tmp_pred = preds[idx]
        idxes_to_merge = [idx]

        prev_sent_idx, prev_sent = book_sentence_idx_search(tmp_pred, tmp_texts)
        while idx + jdx + 1 < len(preds):
            jdx += 1

            tmp_pred = preds[idx + jdx]
            sent_idx, sent = book_sentence_idx_search(tmp_pred, tmp_texts)

            if not sent_idx:
                if debug: print(" [!] NOT FOUND: {}".format(tmp_pred))
                break

            if prev_sent_idx == sent_idx:
                idxes_to_merge.append(idx + jdx)
            else:
                break

        new_keys = get_continuous_audio_paths(
                sum([keys[jdx] for jdx in idxes_to_merge], []))
        text_matches.append([ [tmp_texts[prev_sent_idx]], new_keys ])

        if len(new_keys) > 1:
            book_cursor_idx += 1

        book_cursor_idx = max(book_cursor_idx, sent_idx)

        if idx == len(preds) - 1:
            break
        idx = idx + jdx

    # Counter([len(i) for i in text_matches.values()])
    return text_matches

def get_text_from_audio_batch(paths, multi_process=False):
    results = {}
    items = parallel_run(get_text_from_audio, paths,
            desc="get_text_from_audio_batch")
    for item in items:
        results.update(item)
    return results

def get_text_from_audio(path):
    error_count = 0

    txt_path = path.replace('flac', 'txt')

    if os.path.exists(txt_path):
        with open(txt_path) as f:
            out = json.loads(open(txt_path).read())
            return out

    out = {}
    while True:
        try:
            client = speech.SpeechClient()

            with io.open(path, 'rb') as audio_file:
                content = audio_file.read()
                audio = types.RecognitionAudio(content=content)

            config = types.RecognitionConfig(
                encoding=enums.RecognitionConfig.AudioEncoding.FLAC,
                sample_rate_hertz=16000,
                language_code='ko-KR')

            response = client.recognize(config, audio)
            if len(response.results) > 0:
                alternatives = response.results[0].alternatives

                results = [alternative.transcript for alternative in alternatives]
                assert len(results) == 1, "More than 1 results: {}".format(results)

                out = { path: "" if len(results) == 0 else results[0] }
                print(results[0])
                break
            break
        except:
            error_count += 1
            print("Skip warning for {} for {} times". \
                    format(path, error_count))

            if error_count > 5:
                break
            else:
                continue

    with open(txt_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--asset-dir', type=str, default='assets')
    parser.add_argument('--data-dir', type=str, default='audio')
    parser.add_argument('--pattern', type=str, default="audio/*.flac")
    parser.add_argument('--metadata', type=str, default="metadata.json")
    config, unparsed = parser.parse_known_args()

    paths = glob(config.pattern)
    paths.sort()
    paths = paths

    book_ids = list(set([
            os.path.basename(path).split('.', 1)[0] for path in paths]))
    book_ids.sort()

    def get_finished_ids():
        finished_paths = glob(os.path.join(
                config.asset_dir, "speech-*.json"))
        finished_ids = list(set([
                os.path.basename(path).split('.', 1)[0].replace("speech-", "") for path in finished_paths]))
        finished_ids.sort()
        return finished_ids

    finished_ids = get_finished_ids()

    print("# Finished : {}/{}".format(len(finished_ids), len(book_ids)))

    book_ids_to_parse = list(set(book_ids) - set(finished_ids))
    book_ids_to_parse.sort()

    assert os.path.exists(config.asset_dir), "assert_dir not found"

    pbar = tqdm(book_ids_to_parse, "[1] google_speech",
            initial=len(finished_ids), total=len(book_ids))

    for book_id in pbar:
        current_paths = glob(config.pattern.replace("*", "{}.*".format(book_id)))
        pbar.set_description("[1] google_speech : {}".format(book_id))

        results = get_text_from_audio_batch(current_paths)

        filename = "speech-{}.json".format(book_id)
        path = os.path.join(config.asset_dir, filename)

        with open(path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    finished_ids = get_finished_ids()

    for book_id in tqdm(finished_ids, "[2] text_match"):
        filename = "speech-{}.json".format(book_id)
        path = os.path.join(config.asset_dir, filename)
        clean_path = path.replace("speech", "clean-speech")

        if os.path.exists(clean_path):
            print(" [*] Skip {}".format(clean_path))
        else:
            results = replace_pred_with_book(path)
            with open(clean_path, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    # Dummy

    if False:
        match_paths = get_paths_by_pattern(
                config.asset_dir, 'clean-speech-*.json')

        metadata_path = os.path.join(config.data_dir, config.metadata)

        print(" [3] Merge clean-speech-*.json into {}".format(metadata_path))

        merged_data = []
        for path in match_paths:
            with open(path) as f:
                merged_data.extend(json.loads(f.read()))

        import ipdb; ipdb.set_trace() 

        with open(metadata_path, 'w') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
