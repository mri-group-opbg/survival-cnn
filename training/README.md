
Input images must be mounted in directory /images

Dataframe with list of subjects must be in /data/subjects.pickle


# Docker

## Build

```
docker build . -t tagliente
```

## Run image

```
docker run tagliente python3 /app/training.py --batch-size 1000
```

## Dev mode

Enter interactively in container with:

```
docker run -it -v $(pwd):/app -w /app tagliente
```

# Mounting data

If you want to mount a directory (or a file) you need to to the following:

```
-v /local/path/to/images:/images
```

Bare in mind that first path is local, the second is in the container

