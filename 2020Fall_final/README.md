# Final Project

> Machine Learning Techniques (NTU, Fall 2020) 													instructor: Hsuan-Tien Lin

## Instruction

- file **project.pdf**

## Environment

- OS: Linux
- Python package
  - torch 1.7.1
  - pandas 1.1.5
  - xgboost 1.3.1
  - seaborn, matplotlib

## Execution

- Training model, files are saved in default folder ckpt / img

```bash
python3 NN.py --train [--is_canceled/adr] [--oneHot]
python3 XGB.py --train [--is_canceled/adr] [--oneHot]
python3 randomForest.py --train [--is_canceled/adr] [--oneHot]
python3 XGBRF.py --train [--is_canceled/adr] [--oneHot]
```

- Example: train a one-hot encoding 'is_canceled' XGBoost model

```bash
python3 XGB.py --train --is_canceled --oneHot
```

- Output prediction CSV file
  - parameter example: XGB_oneHot, NN, etc.

```bash
python3 test.py --is_canceled [parameters] --adr [parameters] -output [fileName.csv]
```

- Ensemble

```bash
python3 ensemble.py -input [file1.csv]=[file2.csv]=...=[fileN.csv] -output [bestN.csv]
```

