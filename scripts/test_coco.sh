. ./CONFIG

python run_test.py --images_root $COCO_IMAGES_ROOT \
                    --annotations_root $COCO_CAPTIONS_ROOT \
                    --dataset coco \
                    --coco_set test \
                    --keyword_file $COCO_KEYWORDS_FILE \
                    --fasttext_model $FASTTEXT_EMBEDDING \
                    --pretrained_model $PRETRAINED_COCO_G \
                    --max_no_words $MAX_NO_WORDS 