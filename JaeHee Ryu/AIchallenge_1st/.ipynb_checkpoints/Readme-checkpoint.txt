사용 라이브러리

tqdm
konlpy (Mecab)

학습 커맨드
$ python src/main.py --mode=train --train_file=dataset/train.json \
                   --dev_file=dataset/validate.json \
                   --test_file=dataset/test.json \
                   --batch_size=16

Test 커맨드
$ python src/main.py --mode=test --processed_data \
 --resume=output/model_best.pth.tar --batch_size=1

