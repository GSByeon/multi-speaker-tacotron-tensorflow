#!/bin/sh

CUDA_VISIBLE_DEVICES= python app.py --load_path logs/deepvoice2-256-256-krbook-bah-mon-22000-no-priority --dataname=krbook --num_speakers=1
CUDA_VISIBLE_DEVICES= python app.py --load_path logs/jtbc_2017-09-25_11-49-23 --dataname=krbook --num_speakers=1 --port=5002
CUDA_VISIBLE_DEVICES= python app.py --load_path logs/krbook_2017-09-27_17-02-44 --dataname=krbook --num_speakers=1 --port=5001
CUDA_VISIBLE_DEVICES= python app.py --load_path logs/krfemale_2017-10-10_20-37-38 --dataname=krbook --num_speakers=1 --port=5003
CUDA_VISIBLE_DEVICES= python app.py --load_path logs/krmale_2017-10-10_17-49-49 --dataname=krbook --num_speakers=1 --port=5005
CUDA_VISIBLE_DEVICES= python app.py --load_path logs/park+moon+krbook_2017-10-09_20-43-53 --dataname=krbook --num_speakers=3 --port=5004
