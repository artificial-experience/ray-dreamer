echo "Building Dockerfile with image name dreamer-marl-engineering:1.0"
docker build --no-cache -f docker/Dockerfile -t dreamer-marl-engineering:1.0 .
docker run -it --rm dreamer-marl-engineering:1.0
