# space-invaders-tensorflow
Space Invaders game playing agent with TensorFlow, using openAi Gym's Atari 2600 Space Invaders environment

<img src="https://www.gymlibrary.dev/_images/space_invaders.gif">

## Installation

PIP install the following packages, note **gym** has been downgraded to get this environmant to run properly. 

```text
!pip install -I gym==0.17.3
!pip install tensorflow
!pip install keras
!pip install keras-rl2
!pip install atari-py
!pip install autorom
```

Atari ROMS no longer come packaged with gym, download and install as follows

```text
!mkdir sample_data/ROMS
!AutoROM --install-dir sample_data/ROMS --accept-license
!python -m atari_py.import_roms sample_data/ROMS
```

