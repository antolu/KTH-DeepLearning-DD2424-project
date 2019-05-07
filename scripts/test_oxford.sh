. ./CONFIG

python run_test.py --images_root $OXFORD_IMAGES_ROOT \
                    --annotations_root $OXFORD_CAPTIONS_ROOT \
                    --dataset oxford \
                    --fasttext_model $FASTTEXT_EMBEDDING \
                    --pretrained_model $PRETRAINED_OXFORD_G \
                    --max_no_words $MAX_NO_WORDS 