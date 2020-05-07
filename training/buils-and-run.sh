#!/bin/bash

# Compile the container
docker build . -t tagliente

# Run the container
docker run \
	--rm \
	--name cnn-training \
	--gpus all \
	-v /data/RMN/LUCA_PASQUINI/DATI_SEGMENTATI_SCALATI_media:/images \
	-v /home/user02/notebooks/DataFrame.csv:/config/CSV/DataFrame.csv \
	tagliente \
		python3 /app/training.py \
			--learning-rate=1e-5 \
			--batch-size=16 \
			--dim1=192 \
			--dim2=256 \
			--dim3=144 \
			--sequence-1=T1_registered \
			--sequence-2=T1 \
			--mask-path=SOLID \
			--p=0.8 \
			--kernel-size 3 3 \
			--n-classes=2 \
			--filters=16 \
			--epochs=100 
