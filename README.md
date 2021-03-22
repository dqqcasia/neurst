# End-to-end Speech Translation

This repository is the official implementation of the following papers:

- ["Listen, Understand and Translate": Triple Supervision Decouples End-to-end Speech-to-text Translation](https://arxiv.org/abs/2009.09704)

   *Qianqian Dong, Rong Ye, Mingxuan Wang, Hao Zhou, Shuang Xu, Bo Xu, Lei Li. AAAI 2021.*
   
- [Consecutive Decoding for Speech-to-text Translation](https://arxiv.org/abs/2009.09737)

  *Qianqian Dong, Mingxuan Wang, Hao Zhou, Shuang Xu, Bo Xu, Lei Li. AAAI 2021.*


## Requirements
- Python 3
- Tensorflow 1.15
- Required packages are listed [here](requirements.txt).

To install requirements:

```setup
pip install -r requirements.txt
```

## Data Preprocessing

#### LUT
```preprocess
python3 -m st.tools.dataset configs_template/lut.yaml
```

#### COSTT
```Preprocess
python3 -m st.tools.dataset configs_template/costt.yaml
```

## Train

#### LUT
```train
python3 -m st.bin.run_lut -m train -c configs_template/lut.yaml
```

#### COSTT
```train
python3 -m st.bin.run_costt -m train -c configs_template/costt.yaml
```

## Decode

#### LUT
```decode
python3 -m st.bin.run_lut -m infer -c configs_template/lut.yaml
```

#### COSTT
```decode
python3 -m st.bin.run_costt -m infer -c configs_template/costt.yaml
```

## Citations
Please consider citing our papers in your publications if the project helps your research. BibTeX reference is as follows.
```
@inproceedings{dong2021listen,
  title={Listen, Understand and Translate: Triple Supervision Decouples End-to-end Speech-to-text Translation},
  author={Qianqian Dong, Rong Ye, Mingxuan Wang, Hao Zhou, Shuang Xu, Bo Xu, Lei Li},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2021}
}
```

```
@inproceedings{dong2021consecutive,
  title={Consecutive Decoding for Speech-to-text Translation},
  author={Qianqian Dong, Mingxuan Wang, Hao Zhou, Shuang Xu, Bo Xu, Lei Li},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2021}
}
```