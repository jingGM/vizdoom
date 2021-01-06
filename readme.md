```
sudo apt install cmake libboost-all-dev libsdl2-dev libfreetype6-dev libgl1-mesa-dev libglu1-mesa-dev libpng-dev libjpeg-dev libbz2-dev libfluidsynth-dev libgme-dev libopenal-dev zlib1g-dev timidity tar nasm wget

pip install -r requirements

python train.py -alg "sac" -env 0
python train.py -alg "sac" -env 1
python train.py -alg "ppo" -env 0
python train.py -alg "ppo" -env 1
```