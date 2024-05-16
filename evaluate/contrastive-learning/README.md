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

## Re-produce
1. git clone
```bash
git clone -b eval/contrastive-learning https://github.com/rustic-snob/AcapeLlama.git
git checkout eval/contrastive-learning
```
2. docker
3. tmux
```bash
# If you will use 'tmux'
apt-get update
apt-get install tmux # y
tmux new
```
4. python main.py using bash file
```bash
# run exp 1,2,3
bash scripts/@@@.sh
# wandb option
- 2 enter
- wandb.ai > login > User settings(upper right) > Danger Zone > API keys > Reveal > Copy
- Paste enter
```