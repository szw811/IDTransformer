# IDTransformer

## Train
Download complete training and testing dataset in this [site](https://pan.baidu.com/s/1gAbFRrP80SO52w3BRDyh8g). Code：j4bn.
***
Run 
```
  python train_dn.py --opt=options/train_idtransformer.json
```

## Evaluation
Download trained models and complete testing dataset in thie [site](https://pan.baidu.com/s/1gAbFRrP80SO52w3BRDyh8g). Code：j4bn.

Setting up the following directory structure:

    .
    ├── model_zoo                   
    |   ├──idtransformer_15.pth         # noisy level 15
    |   ├──idtransformer_25.pth         # noisy level 25
    |   |——idtransformer_50.pth          # noisy level 50
    
***
Run 
```
  python my_main_test_idtransformer.py
```

## Acknowledgement
Thanks to [Kai Zhang](https://scholar.google.com.hk/citations?user=0RycFIIAAAAJ&hl) for his work.
