# Dial-QP: A Multi-Tasking and Keyword-Guided Approach for Enhancing Conversational Query Production

The codes of our NLPCC2023 paper: "Dial-QP: A Multi-Tasking and Keyword-Guided Approach for Enhancing Conversational Query Production". 
It can be found here: [paper](https://link.springer.com/chapter/10.1007/978-3-031-44693-1_65).

## Release Timeline
* 2023/10/22 Update dataset and codes.
* 2023/07/30 Initial version.

## Preprocessed Datasets
We conducted experiments on two datasets: Chinese [DuSinc](https://aistudio.baidu.com/datasetdetail/139431), and English [WoI](https://parl.ai/).

We build our datasets by converting output format, extracting dialogue context and dialogue query, and splitting if no splits are given.

Note that except for the dialogue context in the original datasets, our Dial-QP model also takes some extra information as input, including:
* **Hinting keywords**: We use the [SIFRank](https://github.com/sunyilgdx/SIFRank), a unsupervised keyphrase extraction model, to extract contextual hinting keywords from the dialogue context to guide .
* **Query type**: As described in Section 3.1 in the paper, we divide dialogue query into different categories, so we provide category labels in the `cls_3` field.

Finally, we provide the processed datasets in `data/`.

## Run the Code
To train and evaluate the Dial-QP model on the Chinese DuSinc dataset, run the following command:
```bash
python train_Dial-QP.py --data_path ./data/DuSinc.json --model_name fnlp/bart-base-chinese \
--log_file ./log/dusinc.log --save_path ./model/dusinc/ \
--learning_rate 5.5e-5 --batch_size 32 --warmup_steps 400 --log_steps 400 --device cuda:2

python train_Dial-QP.py --infer --data_path ./data/DuSinc.json --model_name fnlp/bart-base-chinese \
--log_file ./log/dusinc.log  --infer_result_path ./infer_res/dusinc_infer_res.txt  \
--batch_size 32 --infer_model_path ${your_trained_model_path}
```
To train and evaluate the Dial-QP model on the English WoI dataset, run the following command:
```bash
python train_Dial-QP.py --is_woi --data_path ./data/WoI.json --model_name facebook/bart-base \
--log_file ./log/woi.log --save_path ./model/woi/ \
--batch_size 16 --warmup_steps 400 --learning_rate 5.5e-5 --log_steps 1000 --device cuda:3

python train_Dial-QP.py --infer --is_woi --data_path ./data/WoI.json --model_name facebook/bart-base \
--log_file ./log/woi.log --infer_result_path ./infer_res/woi_infer_res.txt \
--batch_size 16 --infer_model_path ${your_trained_model_path}
```

## Citation
If you find this repository helpful, please kindly cite the paper.
````
@inproceedings{yu2023dial,
  title={Dial-QP: A Multi-tasking and Keyword-Guided Approach for Enhancing Conversational Query Production},
  author={Yu, Jiong and Wu, Sixing and Wang, Shuoxin and Lai, Haosen and Zhou, Wei},
  booktitle={CCF International Conference on Natural Language Processing and Chinese Computing},
  pages={840--852},
  year={2023},
  organization={Springer}
}
````