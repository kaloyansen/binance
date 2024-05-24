# quick setup

git clone git@github.com:kaloyansen/binance 

BINANCE_KEY=your-api-key

BINANCE_SECRET=your-api-secret

python -m venv apienv

source apienv/bin/activate 

pip install -r binance/requirements.txt

cd binance

./live.py


