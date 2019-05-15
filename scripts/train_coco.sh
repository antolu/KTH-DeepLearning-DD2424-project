. ./CONFIG

python run_test.py --images_root $COCO_IMAGES_ROOT \
                    --annotations_root $COCO_CAPTIONS_ROOT \
                    --dataset coco \
                    --coco_set train \
                    --keywords_file $COCO_KEYWORDS_FILE \
                    --fasttext_model $FASTTEXT_EMBEDDING \
                    --pretrained_generator $PRETRAINED_COCO_G \
                    --pretrained_discriminator $PRETRAINED_COCO_D \
                    --max_no_words $MAX_NO_WORDS \
                    --blacklist $BLACKLIST \
                    --runtype train \
		    --no_epochs 600
