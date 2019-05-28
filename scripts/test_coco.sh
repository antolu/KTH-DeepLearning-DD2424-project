. ./CONFIG

python run.py --images_root $COCO_IMAGES_ROOT \
              --annotations_root $COCO_CAPTIONS_ROOT \
              --dataset coco \
              --coco_set train \
              --keywords_file $COCO_KEYWORDS_FILE \
              --fasttext_model $FASTTEXT_EMBEDDING \
              --pretrained_generator $PRETRAINED_COCO_G \
              --blacklist $BLACKLIST \
              --mode test \
              --no_epochs 600
