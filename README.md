## EGICH
The code for **[External Guidance Incomplete Cross-modal Hashing (TIP'26)](https://ieeexplore.ieee.org/abstract/document/11433515)**

## Code Repository Directory
`EGICH/`  
`├── clip/` &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # get from **[clip](https://huggingface.co/openai/clip-vit-base-patch32)** `
`│   ├── clip.py`  
`│   ├── config.json`  
`│   ├── preprocessor_config.json`  
`│   ├── pytorch_model.bin`  
`│   └── tokenizer.json`  
`├── utils/`  
`│   └── calc_hammingranking.py`  
`├── wordnet/`  
`│   ├── wordnet_embedding.py`  
`│   └── WordNetNouns.csv`  
`├── load_dataset.py`  
`├── loss.py`  
`├── main.py`  
`├── models.py`  
`├── ops.py`  
`├── settings.py`  
`└── trainer.py`

## RUN
```bash
python3 wordnet/wordnet_embedding.py
python3 main.py
