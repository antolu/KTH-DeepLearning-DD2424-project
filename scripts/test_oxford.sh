. ./CONFIG

python run.py --images_root $OXFORD_IMAGES_ROOT \
              --annotations_root $OXFORD_CAPTIONS_ROOT \
              --dataset oxford \
              --fasttext_model $FASTTEXT_EMBEDDING \
              --pretrained_generator $PRETRAINED_OXFORD_G \
              --mode test
