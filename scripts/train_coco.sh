. ./CONFIG

python run.py --images_root $COCO_IMAGES_ROOT \
              --annotations_root $COCO_CAPTIONS_ROOT \
              --dataset coco \
              --coco_set train \
              --keywords_file $COCO_KEYWORDS_FILE \
              --fasttext_model $FASTTEXT_EMBEDDING \
              --blacklist $BLACKLIST \
              --pretrained_generator $PRETRAINED_COCO_G \
              --pretrained_discriminator $PRETRAINED_COCO_D \
              --pretrained_optimizer_discriminator $PRETRAINED_COCO_OD \
              --pretrained_optimizer_generator $PRETRAINED_COCO_OG \
              --mode train \
              --learning_rate 0.0002 \
              --batch_size 64 \
              --no_epochs 600 \
              --momentum 0.5 \
              --save_models_frequency 50 \
              --lambda_1 10 \
              --lambda_2 2.0
