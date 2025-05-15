python train.py --dataset CUSTOM --path ./CUSTOM_P12_Q5/ --P 12 --Q 5 --L 1 --time_slot 15 --adjdata CUSTOM_P12_Q5/adj_mx.pkl

python test.py   --P 12   --Q 5   --save_path ./CUSTOM_P12_Q5  --se_file ./CUSTOM_P12_Q5/SE(CUSTOM).txt

python train.py --dataset CUSTOM --path ./CUSTOM_P12_Q1/ --P 12 --Q 1 --L 1 --time_slot 15 --adjdata CUSTOM_P12_Q1/adj_mx.pkl

python train.py --dataset CUSTOM --path ./CUSTOM_P12_Q1/ --P 12 --Q 1 --L 2 --time_slot 15 --adjdata CUSTOM_P12_Q1/adj_mx.pkl

python test.py   --P 12 --Q 1   --save_path ./CUSTOM_P20_Q1   --se_file ./SE(CUSTOM).txt

python test.py   --P 20 --Q 1   --save_path ./CUSTOM_P20_Q1   --se_file ./CUSTOM_P20_Q1/SE(CUSTOM).txt

python train.py --dataset CUSTOM --path ./CUSTOM_P20_Q1/ --P 20 --Q 1 --L 1 --time_slot 15 --adjdata CUSTOM_P20_Q1/adj_mx.pkl

python test.py   --P 12 --Q 1   --save_path ./CUSTOM_P12_Q1   --se_file ./CUSTOM_P12_Q1/SE(CUSTOM).txt

python train.py --dataset CUSTOM --path ./CUSTOM_P12_Q1/ --P 12 --Q 1 --L 1 --time_slot 15 --adjdata CUSTOM_P12_Q1/adj_mx.pkl

python train.py --dataset CUSTOM --path ./CUSTOM_P12_Q1/ --P 12 --Q 1 --L 2 --time_slot 15 --adjdata CUSTOM_P12_Q1/adj_mx.pkl
