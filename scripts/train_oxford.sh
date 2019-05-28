. ./CONFIG

python run.py --images_root $OXFORD_IMAGES_ROOT \
              --annotations_root $OXFORD_CAPTIONS_ROOT \
              --dataset oxford \
              --fasttext_model $FASTTEXT_EMBEDDING \
              --blacklist $BLACKLIST \
#              --pretrained_generator $PRETRAINED_OXFORD_G \
#              --pretrained_discriminator $PRETRAINED_OXFORD_D \
#              --pretrained_optimizer_discriminator $PRETRAINED_OXFORD_OD \
#              --pretrained_optimizer_generator $PRETRAINED_OXFORD_OG \
              --mode train \
              --learning_rate 0.0002 \
              --batch_size 64 \
              --no_epochs 600 \
              --momentum 0.5 \
              --save_models_frequency 50 \
              --lambda_1 10 \
              --lambda_2 2.0
