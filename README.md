## EGICH
The code for **[External Guidance Incomplete Cross-modal Hashing (TIP'26)](https://ieeexplore.ieee.org/abstract/document/11433515)**.

## Code Repository Directory
Get the files of clip from [clip](https://huggingface.co/openai/clip-vit-base-patch32).
```text
EGICH/
├── clip/                          
│   ├── clip.py 
│   ├── config.json 
│   ├── preprocessor_config.json
│   ├── pytorch_model.bin      
│   └── tokenizer.json
├── dataset/                        
│   ├── MIRFlickr.h5
│   ├── MS-COCO.h5
│   └── NUS-WIDE.h5
├── utils/                        
│   └── calc_hammingranking.py          
├── wordnet/ 
│   ├── wordnet_embedding.py
│   └── WordNetNouns.csv      
├── load_dataset.py        
├── loss.py        
├── main.py      
├── models.py            
├── ops.py  
├── settings.py  
└── trainer.py
```

## Environment Setup
Python Version: 3.8.10

PyTorch (CUDA 11.8):
```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install transformers pandas
```

## RUN 
First, run the following code to generate `wordnet_embedding_ensemble.npy`.
```bash
python3 wordnet/wordnet_embedding.py
```
Then, run the following code to perform training.
```bash
python3 main.py
```
