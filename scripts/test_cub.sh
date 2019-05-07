. ./CONFIG

python run_test.py --images_root $CUB_IMAGES_ROOT \
                    --annotations_root $CUB_CAPTIONS_ROOT \
                    --dataset cub \
                    --fasttext_model $FASTTEXT_EMBEDDING \
                    --pretrained_model $PRETRAINED_CUB_G \
                    --max_no_words $MAX_NO_WORDS 