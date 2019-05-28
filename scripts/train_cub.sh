. ./CONFIG

python run.py --images_root $CUB_IMAGES_ROOT \
              --annotations_root $CUB_CAPTIONS_ROOT \
              --dataset cub \
              --fasttext_model $FASTTEXT_EMBEDDING \
              --blacklist $BLACKLIST \
#              --pretrained_generator $PRETRAINED_CUB_G \
#              --pretrained_discriminator $PRETRAINED_CUB_D \
#              --pretrained_optimizer_discriminator $PRETRAINED_CUB_OD \
#              --pretrained_optimizer_generator $PRETRAINED_CUB_OG \
              --mode train \
              --learning_rate 0.0002 \
              --batch_size 64 \
              --no_epochs 600 \
              --momentum 0.5 \
              --save_models_frequency 50 \
              --lambda_1 10 \
              --lambda_2 2.0
