. ./CONFIG

python run_test.py --images_root $CUB_IMAGES_ROOT \
                    --annotations_root $CUB_CAPTIONS_ROOT \
                    --dataset cub \
                    --fasttext_model $FASTTEXT_EMBEDDING \
                    --blacklist $BLACKLIST \
                    --max_no_words $MAX_NO_WORDS  \
                    --runtype train \
                    --batch_size 64 \
                    --no_epochs 600
#		    --pretrained_generator models/run_G_dataset_cub_before_dying.pth \
#	 	    --pretrained_discriminator models/run_D_dataset_cub_before_dying.pth
