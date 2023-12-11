## [CMU NLP 23f team project] What the DAAM : Interpreting Stable Diffusion Using Cross Attention

### Member
- Mijoo Kim, Yejin Kim, Minwoo Kang, Sungju Oh, Wonjin Yang

### Description
* Implentation of [what the DAAM : Interpreting Stable Diffusion Using Cross Attention, NACL'23](https://arxiv.org/abs/2210.04885)
* official repository : [github](https://github.com/castorini/daam)


### Environment setup
```bash
pip install -r requirements.txt
```

```bash
python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('YOUR TOKEN')"
```

### Run Demo

```bash
python main.py --prompt 'Two dogs run across the field' --word 'dogs'  
```

The result will be saved in `./result`
