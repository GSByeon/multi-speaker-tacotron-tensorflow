import os
import re
import math
import argparse
from glob import glob

from synthesizer import Synthesizer
from train import create_batch_inputs_from_texts
from utils import makedirs, str2bool, backup_file
from hparams import hparams, hparams_debug_string


texts = [
    '텍스트를 음성으로 읽어주는 "음성합성" 기술은 시각 장애인을 위한 오디오북, 음성 안내 시스템, 대화 인공지능 등 많은 분야에 활용할 수 있습니다.',
    "하지만 개인이 원하는 목소리로 음성합성 엔진을 만들기에는 어려움이 많았고 소수의 기업만이 기술을 보유하고 있었습니다.",
    "최근 딥러닝 기술의 발전은 음성합성 기술의 진입 장벽을 많이 낮췄고 이제는 누구나 손쉽게 음성합성 엔진을 만들 수 있게 되었습니다.",

    "본 세션에서는 딥러닝을 활용한 음성합성 기술을 소개하고 개발 경험과 그 과정에서 얻었던 팁을 공유하고자 합니다.",
    "음성합성 엔진을 구현하는데 사용한 세 가지 연구를 소개하고 각각의 기술이 얼마나 자연스러운 목소리를 만들 수 있는지를 공유합니다.",

    # Harry Potter
    "그리고 헤르미온느는 겁에 질려 마룻바닥에 쓰러져 있었다.",
    "그러자 론은 요술지팡이를 꺼냈다. 무엇을 할지도 모르면서 그는 머리에 처음으로 떠오른 주문을 외치고 있었다.",
    "윙가르디움 레비오우사.... 하지만, 그렇게 소리쳤다.",
    "그러자 그 방망이가 갑자기 트롤의 손에서 벗어나, 저 위로 올라가더니 탁하며 그 주인의 머리 위에 떨어졌다.",
    "그러자 트롤이 그 자리에서 비틀거리더니 방 전체를 흔들어버릴 것 같은 커다란 소리를 내며 쿵 하고 넘어졌다. ",
    "그러자 조그맣게 펑 하는 소리가 나면서 가장 가까이 있는 가로등이 꺼졌다.",
    "그리고 그가 다시 찰깍하자 그 다음 가로등이 깜박거리며 나가 버렸다.",

    #"그가 그렇게 가로등 끄기를 열두번 하자, 이제 그 거리에 남아 있는 불빛이라곤, ",
    #"바늘로 꼭 질러둔 것처럼 작게 보이는 멀리서 그를 지켜보고 있는 고양이의 두 눈뿐이었다.",
    #"프리벳가 4번지에 살고 있는 더즐리 부부는 자신들이 정상적이라는 것을 아주 자랑스럽게 여기는 사람들이었다. ",
    #"그들은 기이하거나 신비스런 일과는 전혀 무관해 보였다.",
    #"아니, 그런 터무니없는 것은 도저히 참아내지 못했다.",
    #"더즐리 씨는 그루닝스라는 드릴제작회사의 중역이었다.",
    #"그는 목이 거의 없을 정도로 살이 뒤룩뒤룩 찐 몸집이 큰 사내로, 코밑에는 커다란 콧수염을 기르고 있었다.",
    #"더즐리 부인은 마른 체구의 금발이었고, 목이 보통사람보다 두 배는 길어서, 담 너머로 고개를 쭉 배고 이웃 사람들을 몰래 훔쳐보는 그녀의 취미에는 더없이 제격이었다.",

    # From Yoo Inna's Audiobook (http://campaign.happybean.naver.com/yooinna_audiobook):
    #'16세기 중엽 어느 가을날 옛 런던 시의 가난한 캔티 집안에 사내아이 하나가 태어났다.',
    #'그런데 그 집안에서는 그 사내아이를 별로 반기지 않았다.',
    #'바로 같은 날 또 한 명의 사내아이가 영국의 부유한 튜터 가문에서 태어났다.',
    #'그런데 그 가문에서는 그 아이를 무척이나 반겼다.',
    #'온 영국이 다 함께 그 아이를 반겼다.',

    ## From NAVER's Audiobook (http://campaign.happybean.naver.com/yooinna_audiobook):
    #'부랑자 패거리는 이른 새벽에 일찍 출발하여 길을 떠났다.',
    #'하늘은 찌푸렸고, 발밑의 땅은 질퍽거렸으며, 겨울의 냉기가 공기 중에 감돌았다.',
    #'지난밤의 흥겨움은 온데간데없이 사라졌다.',
    #'시무룩하게 말이 없는 사람들도 있었고, 안달복달하며 조바심을 내는 사람들도 있었지만, 기분이 좋은 사람은 하나도 없었다.',

    ## From NAVER's nVoice example (https://www.facebook.com/naverlabs/videos/422780217913446):
    #'감사합니다. Devsisters 김태훈 님의 발표였습니다.',
    #'이것으로 금일 마련된 track 2의 모든 세션이 종료되었습니다.',
    #'장시간 끝까지 참석해주신 개발자 여러분들께 진심으로 감사의 말씀을 드리며,',
    #'잠시 후 5시 15분부터 특정 주제에 관심 있는 사람들이 모여 자유롭게 이야기하는 오프미팅이 진행될 예정이므로',
    #'참여신청을 해주신 분들은 진행 요원의 안내에 따라 이동해주시기 바랍니다.',

    ## From Kakao's Son Seok hee example (https://www.youtube.com/watch?v=ScfdAH2otrY):
    #'소설가 마크 트웨인이 말했습니다.',
    #'인생에 가장 중요한 이틀이 있는데, 하나는 세상에 태어난 날이고 다른 하나는 왜 이 세상에 왔는가를 깨닫는 날이다.',
    #'그런데 그 첫번째 날은 누구나 다 알지만 두번째 날은 참 어려운 것 같습니다.',
    #'누구나 그 두번째 날을 만나기 위해 애쓰는게 삶인지도 모르겠습니다.',
    #'뉴스룸도 그런 면에서 똑같습니다.',
    #'저희들도 그 두번째의 날을 만나고 기억하기 위해 매일 매일 최선을 다하겠습니다.',
]


def get_output_base_path(load_path, eval_dirname="eval"):
    if not os.path.isdir(load_path):
        base_dir = os.path.dirname(load_path)
    else:
        base_dir = load_path

    base_dir = os.path.join(base_dir, eval_dirname)
    if os.path.exists(base_dir):
        backup_file(base_dir)
    makedirs(base_dir)

    m = re.compile(r'.*?\.ckpt\-([0-9]+)').match(load_path)
    base_path = os.path.join(base_dir,
            'eval-%d' % int(m.group(1)) if m else 'eval')
    return base_path


def run_eval(args):
    print(hparams_debug_string())

    load_paths = glob(args.load_path_pattern)

    for load_path in load_paths:
        if not os.path.exists(os.path.join(load_path, "checkpoint")):
            print(" [!] Skip non model directory: {}".format(load_path))
            continue

        synth = Synthesizer()
        synth.load(load_path)

        for speaker_id in range(synth.num_speakers):
            base_path = get_output_base_path(load_path, "eval-{}".format(speaker_id))

            inputs, input_lengths = create_batch_inputs_from_texts(texts)

            for idx in range(math.ceil(len(inputs) / args.batch_size)):
                start_idx, end_idx = idx*args.batch_size, (idx+1)*args.batch_size

                cur_texts = texts[start_idx:end_idx]
                cur_inputs = inputs[start_idx:end_idx]

                synth.synthesize(
                        texts=cur_texts,
                        speaker_ids=[speaker_id] * len(cur_texts),
                        tokens=cur_inputs,
                        base_path="{}-{}".format(base_path, idx),
                        manual_attention_mode=args.manual_attention_mode,
                        base_alignment_path=args.base_alignment_path,
                )

        synth.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=16)
    parser.add_argument('--load_path_pattern', required=True)
    parser.add_argument('--base_alignment_path', default=None)
    parser.add_argument('--manual_attention_mode', default=0, type=int,
            help="0: None, 1: Argmax, 2: Sharpening, 3. Pruning")
    parser.add_argument('--hparams', default='',
        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    args = parser.parse_args()

    #hparams.max_iters = 100
    #hparams.parse(args.hparams)
    run_eval(args)


if __name__ == '__main__':
    main()
