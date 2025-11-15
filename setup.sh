mkdir -p data/raw
mkdir -p data/processed
python main.py --mode download
python main.py --mode preprocess
