# dpr_for_vision

## Tree
```python
├─config
│      config.json # train setting
│      
├─dataset
│      dataset.json # data information
│      train_dataset.json 
│      val_dataset.json
│      
├─model
│      dpr.py # model
│      
├─scripts
│      main.py # run train
│      trainer.py # train & val & test process
│      
└─utils
        data_preprocessing.py # dataset class
```

## Usage
```.sh
python scripts/main.py config/config.json
```

데이터셋은 main 데이터셋만을 활용하였으며, {item_id: wearing_image_list} 형식으로 구성하여 wearing image를 키로 사용하도록 설정
