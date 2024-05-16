# train/contrastive_learning
- semantic similarity 측정에 필요한 encoder 학습
- contrastive learning 활용 → 동일 가사 내의 샘플은 가깝게

## File Structure
```html
|-- README.md
|-- main.py
|-- dataset/
|  |-- ...
|-- modules/
|  |-- data.py
|  |-- model.py
|  |-- utils.py
|-- results/
|  |-- model_name/loss_name/batch_size/seed/
|  |-- ...
|-- configs/
|  |-- default.yaml
|  |-- model/
|  |  |-- ...
|  |-- train/
|  |  |-- ...
|-- scripts/
|  |-- default.sh
|  |-- model_exp.sh
|  |-- bsz_exp.sh
```