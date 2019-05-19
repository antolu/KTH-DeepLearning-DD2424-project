. ./CONFIG

python run.py --images_root $OXFORD_IMAGES_ROOT \
                    --annotations_root $OXFORD_CAPTIONS_ROOT \
                    --dataset oxford \
                    --fasttext_model $FASTTEXT_EMBEDDING \
                    --pretrained_generator $PRETRAINED_OXFORD_G \
                    --pretrained_discriminator $PRETRAINED_OXFORD_D \
                    --max_no_words $MAX_NO_WORDS  \
                    --mode test