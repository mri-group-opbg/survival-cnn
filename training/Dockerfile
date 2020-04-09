FROM tensorflow/tensorflow:1.15.2-gpu-py3

# Install libraries
RUN pip install --user nipy nilearn seaborn keras

# Copy source code
ADD ./training.py /app/training.py

# Main command
CMD [ "python3", "/app/training.py" ]

