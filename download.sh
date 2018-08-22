if [ -z "$STORAGE_DIR" ]; then
    echo "STORAGE_DIR is not set, so SQUAD will be downloaded to './data'."
    STORAGE_DIR="."
fi

DATASET_DIR="$STORAGE_DIR/data/squad"
GLOVE_DIR="$STORAGE_DIR/wordvec/glove"
mkdir -p "$DATASET_DIR" "$GLOVE_DIR"

# Download training set and dev set
wget "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json" -O "$DATASET_DIR/train-v1.1.json"
wget "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json" -O "$DATASET_DIR/dev-v1.1.json"

# Download official evaluation script
wget "https://worksheets.codalab.org/rest/bundles/0xbcd57bee090b421c982906709c8c27e1/contents/blob/" -O "official_evaluate.py"

# Download GloVe
wget "http://nlp.stanford.edu/data/glove.840B.300d.zip" -O "$GLOVE_DIR/glove.840B.300d.zip"
unzip "$GLOVE_DIR/glove.840B.300d.zip" -d "$GLOVE_DIR"
rm "$GLOVE_DIR/glove.840B.300d.zip"