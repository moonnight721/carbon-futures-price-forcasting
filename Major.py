import subprocess

model_scripts=[
    "RNN.py",
    "GRU.py",
    "LSTM.py",
    "CNN-GRU.py",
    "CNN-LSTM.py",
    "CNN-RNN.py",
    # "TCNo.py",
    # "LSTM-LSTM-DNN.py"    
]

for scripts in model_scripts:
    print(f"\n 執行模型:{scripts}")
    try:
        subprocess.run(["python", scripts])
    except subprocess.CalledProcessError as e:
        print(f"執行失敗:{scripts}\n{e.stderr}")
