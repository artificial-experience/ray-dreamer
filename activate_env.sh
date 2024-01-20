DIR=$(realpath $(dirname "${BASH_SOURCE[0]}"))
RLROOT=$DIR
export RLROOT

poetry lock
poetry install
source $(poetry env info --path)/bin/activate
echo "Ray Activated"
