#!/bin/bash

# Compile the container
docker build . -t tagliente

# Run the container
docker run \
	-v /data/RMN/LUCA_PASQUINI/DATI_SEGMENTATI_SCALATI_media:/images \
	-v /home/user02/notebooks/DataFrame.csv:/config/CSV/DataFrame.csv \
	tagliente \
		python3 /app/training.py --batch-size 1000


