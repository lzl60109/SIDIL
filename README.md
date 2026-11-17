# Toward Reliable Imitation Learning With Limited Expert Demonstrations via Search-Based Inverse Dynamic Learning

## Expert Policies/Noise Demonstrations

To generate demonstrations, see files `scripts/<TASK>/gen_demos.py` and `scripts/<TASK>//gen_demos.sh`.For example,`scripts/fetch/gen_demos.py` 

## Experiments

To train and evaluate SIDIL, run the following command:
```bash
python experiments/SIDIL.py +experiments=SIDIL/fetch_pick
```

A run script is provided in `scripts/<TASK>//run.sh`. 



## References

<a id="1">[1]</a>
[Robust Imitation of a Few Demonstrations with a Backwards Model](https://arxiv.org/abs/2210.09337), Jung Yeon Park et al, 2022.

## Citation

If you find this work useful, please cite the paper as follows:
```
@article{lin2026toward,
  title={Toward reliable imitation learning with limited expert demonstrations via search-based inverse dynamic learning},
  author={Lin, Zhiliang and Chen, Zhuangzhuang and Zhu, Guanming and Wang, Li and Li, Jianqiang},
  journal={IEEE Transactions on Automation Science and Engineering},
  year={2026},
  publisher={IEEE}
}```
