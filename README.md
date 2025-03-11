# CNN for Weight Prediction

## 🚀 How to Train the Model

### 📌 Default Training
Run the following command to train the model with default settings (100 epochs, batch size 32, learning rate 0.001):
```bash
python train.py
```

### 📌 Custom Training with Command-Line Arguments
You can modify training settings using command-line arguments:
```bash
python train.py --epochs 50 --batch_size 64 --learning_rate 0.0005 \
--data_path ./data/custom_dataset.csv \
--save_state_dict ./saved_models \
--log_dir ./custom_logs
```
- `--epochs` : Number of training epochs (default: 100)
- `--batch_size` : Batch size for training (default: 32)
- `--learning_rate` : Learning rate for optimization (default: 0.001)
- `--data_path` : Path to the dataset CSV file (default: `./data/SOCR-HeightWeight.csv`)
- `--save_state_dict` : Directory to save the trained model (default: `./models`)
- `--log_dir` : Directory for TensorBoard logs (default: `./logs`)

### 📌 Monitor with TensorBoard
During training, you can monitor the loss using TensorBoard:
```bash
tensorboard --logdir=logs
```
Then, open [http://localhost:6006/](http://localhost:6006/) in your browser.

## 📌 Model Evaluation
After training, you can evaluate the model using:
```bash
python src/evaluate.py --data_path ./data/custom_dataset.csv --save_state_dict ./saved_models
```
- `--data_path` : Path to the dataset CSV file (default: `./data/SOCR-HeightWeight.csv`)
- `--save_state_dict` : Directory where the trained model is stored (default: `./models`)

## 📌 Project Structure
- `data/` - Dataset files
- `logs/` - TensorBoard logs
- `models/` - Trained models
- `notebooks/` - Jupyter notebooks for analysis
- `src/` - Source code
  - `dataset.py` - Data preprocessing
  - `model.py` - CNN architecture
  - `train.py` - Training script
  - `evaluate.py` - Model evaluation
  - `utils.py` - Utility functions
- `requirements.txt` - Required packages
- `README.md` - Documentation
- `train.py` - Main script (launcher script for convenience)

## 📁 Folder Structure
```bash
Weight-Prediction-CNN/
│── data/                     # Stores dataset
│── logs/                     # TensorBoard logs
│── models/                   # Trained models
│── notebooks/                # Jupyter notebooks for analysis
│── src/                      # Source code
│   │── dataset.py            # Dataset processing
│   │── model.py              # CNN model definition
│   │── train.py              # Training script
│   │── evaluate.py           # Evaluation script
│   └── utils.py              # Utility functions
│── requirements.txt          # Required packages
│── README.md                 # Documentation
└── train.py                  # Main script (launcher script)
```

