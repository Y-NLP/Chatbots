import json
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import transformers
import sys
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import BertTokenizer, BartForConditionalGeneration,BartTokenizer
from tqdm import tqdm
from rouge import Rouge
# from datasets import load_metric
import argparse
import logging
from torch.cuda.amp import GradScaler, autocast


# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_logger(log_path):
    """
    将日志输出到日志文件和控制台
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')

    # 创建一个handler，用于写入日志文件
    # 增加对logger.info中文乱码的处理
    # python3.9以上才可以设置encoding
    if sys.version_info.major == 3 and sys.version_info.minor >= 9:
        file_handler = logging.FileHandler(filename=log_path, encoding='utf-8')
    else:
        file_handler = logging.FileHandler(filename=log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    # 创建一个handler，用于将日志输出到控制台
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


# 加载数据
def load_data(args, logger):
    logger.info('Loading data from {}'.format(args.data_path))

    if args.with_keywords:
        logger.info(f'keywords已经加入，请确保{args.data_path}中的src中包含keywords')
    if args.with_three_classifier:
        logger.info(f'三分类已经加入，请确保{args.data_path}中的cls_3中包含三分类标签')

    with open(args.data_path, 'r', encoding='utf-8') as f:
        train_data = {'src': [], 'tgt': [], 'cls': [], 'cls_3': []}
        valid_data = {'src': [], 'tgt': [], 'cls': [], 'cls_3': []}
        test_data = {'src': [], 'tgt': [], 'cls': [], 'cls_3': []}
        all_data = json.load(f)
        all_data_train = all_data['train']
        all_data_valid = all_data['valid']
        all_data_test = all_data['test']

        for item in all_data_train:
            train_data['src'].append(item['src'].strip())
            train_data['tgt'].append(item['tgt'].strip())
            train_data['cls'].append(item['cls'])
            if args.with_three_classifier:
                train_data['cls_3'].append(item['cls_3'])
        for item in all_data_valid:
            valid_data['src'].append(item['src'].strip())
            valid_data['tgt'].append(item['tgt'].strip())
            valid_data['cls'].append(item['cls'])
            if args.with_three_classifier:
                valid_data['cls_3'].append(item['cls_3'])
        for item in all_data_test:
            test_data['src'].append(item['src'].strip())
            test_data['tgt'].append(item['tgt'].strip())
            test_data['cls'].append(item['cls'])
            if args.with_three_classifier:
                test_data['cls_3'].append(item['cls_3'])

    logger.info('Loading data finished.')
    logger.info('train_data num: {}'.format(len(train_data['src'])))
    logger.info('valid_data num: {}'.format(len(valid_data['src'])))
    logger.info('test_data num: {}'.format(len(test_data['src'])))

    return train_data, valid_data, test_data


# 定义数据集
class Seq2SeqAndClassificationDataset(Dataset):
    def __init__(self, tokenizer, src_texts, tgt_texts, cls_labels, cls_labels_3, args, max_length=1024):
        self.tokenizer = tokenizer
        self.src_texts = src_texts
        self.tgt_texts = tgt_texts
        self.cls_labels = cls_labels
        self.cls_labels_3 = cls_labels_3
        self.max_length = max_length
        self.args = args

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]
        cls_label = self.cls_labels[idx]
        if self.args.with_three_classifier:
            cls_label_3 = self.cls_labels_3[idx]
            return {"src_text": src_text, "tgt_text": tgt_text, "cls_label": cls_label, "cls_label_3": cls_label_3}
        else:
            return {"src_text": src_text, "tgt_text": tgt_text, "cls_label": cls_label}


# 自定义collate_fn函数
def seq2seq_and_classification_collate_fn(batch, tokenizer, args, max_length=None):
    src_texts = [item["src_text"] for item in batch]
    tgt_texts = [item["tgt_text"] for item in batch]
    cls_labels = [item["cls_label"] for item in batch]
    if args.with_three_classifier:
        cls_labels = [item["cls_label_3"] for item in batch]

    # src和tgt的最大长度
    if max_length is None:
        max_length_src = max(len(src_text) + 2 for src_text in src_texts)
        max_length_tgt = max(len(tgt_text) + 2 for tgt_text in tgt_texts)
        if max_length_src > 1024:
            max_length_src = 1024
        if max_length_tgt > 1024:
            max_length_tgt = 1024
    else:
        max_length_src = max_length
        max_length_tgt = max_length

    src_inputs = tokenizer(src_texts, max_length=max_length_src, padding='max_length', truncation=True,
                           return_tensors='pt')
    tgt_inputs = tokenizer(tgt_texts, max_length=max_length_tgt, padding='max_length', truncation=True,
                           return_tensors='pt')

    input_ids = src_inputs["input_ids"]

    # src_inputs格式：        [CLS]aaa[SEP][PAD][PAD]
    # tgt_inputs格式：        [CLS]bbb[SEP][PAD][PAD]
    # decoder_input_ids格式   [SEP][CLS]bbb[SEP][PAD]
    # labels格式：            [CLS]bbb[SEP][UNK][UNK]
    # 模型输出格式:            [SEP][CLS]bbb[SEP]

    # 将tgt_inputs左移移一位，并在开头添加[SEP] token
    decoder_input_ids = torch.cat((tokenizer.sep_token_id * torch.ones(tgt_inputs["input_ids"].size(0), 1,
                                                                       dtype=torch.long),
                                   tgt_inputs["input_ids"][:, :-1]), dim=1)

    # 将tgt_inputs的padding_id换为-100 (ignore_index for CrossEntropyLoss)
    labels = tgt_inputs["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100

    return {"src": src_inputs,
            "tgt": tgt_inputs,
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "labels": labels,
            "cls": torch.tensor(cls_labels, dtype=torch.long)}


# 训练函数
def train(model, dataloader, optimizer, scheduler, logger, args):
    model.train()
    total_loss = 0
    batch_id = 0
    if not args.not_use_amp:
        logger.info("Use amp.")
        scaler = GradScaler()  # 添加 GradScaler
    model_type = 'BART'
    if args.with_classifier:
        if args.with_three_classifier:
            model_type = model_type + ' + three_classification'
        else:
            model_type = model_type + ' + classification'
    if args.with_keywords:
        model_type = model_type + ' + keywords'
    logger.info(f"Model type: {model_type}")

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(args.device)
        decoder_input_ids = batch["decoder_input_ids"].to(args.device)
        labels = batch["labels"].to(args.device)
        cls_labels = batch["cls"].to(args.device)

        if not args.not_use_amp:
            with autocast():  # 使用 autocast 包装前向传播
                if args.with_classifier:
                    # logger.info("BART+classification")
                    # 计算分类损失
                    encoder_outputs = model.model.encoder(input_ids)
                    classifier_output = model.classifier(encoder_outputs.last_hidden_state[:, 0, :])
                    classification_loss = nn.CrossEntropyLoss()(classifier_output, cls_labels).mean()

                    # 计算生成任务损失
                    if not args.with_three_classifier:
                        # 对于标签为1的样本进行生成任务训练
                        generation_mask = (cls_labels == 1)  # 使用cls_labels而不是classification_preds来获取需要生成的样本
                    else:
                        # 对于标签为1或2的样本进行生成任务训练
                        generation_mask = (cls_labels != 0)
                    encoder_outputs_gen = encoder_outputs[0][generation_mask]
                    decoder_input_ids_gen = decoder_input_ids[generation_mask]
                    labels_gen = labels[generation_mask]

                    if generation_mask.any():  # 如果有需要生成的样本
                        outputs = model(input_ids=None, encoder_outputs=(encoder_outputs_gen,),
                                        attention_mask=None, decoder_input_ids=decoder_input_ids_gen,
                                        labels=labels_gen)
                        generation_loss = outputs.loss.mean()
                    else:
                        generation_loss = torch.tensor(0.0, device=args.device)

                    # 计算总损失
                    loss = args.classification_weight * classification_loss + args.generation_weight * generation_loss
                else:
                    outputs = model.forward(input_ids=input_ids,
                                            decoder_input_ids=decoder_input_ids, labels=labels)
                    loss = outputs.loss.mean()

            scaler.scale(loss).backward()  # 使用 scaler.scale 调整损失
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)  # 使用 scaler.step 更新优化器
            scaler.update()  # 使用 scaler.update 更新缩放器
        else:
            raise NotImplementedError('不使用混合精度训练显存太大')

        scheduler.step()

        total_loss += loss.item()
        batch_id += 1

        if batch_id % args.log_steps == 0:
            logger.info("batch: {}, loss: {}".format(batch_id, loss.item()))
            if args.with_classifier:
                logger.info("avg classification_loss: {}, generation_loss: {}".format(
                    args.classification_weight * classification_loss.item(),
                    args.generation_weight * generation_loss.item()))
            logger.info("lr: {}".format(scheduler.get_last_lr()[0]))

            # 打印数据
            logger.info(f"input_ids: {args.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)}")
            logger.info(f"decoder_input_ids: {args.tokenizer.decode(decoder_input_ids[0].tolist())}")
            if args.is_woi:
                test_labels = torch.where(labels[0]== -100, args.tokenizer.pad_token_id, labels[0]).tolist()
                logger.info(f"labels: {args.tokenizer.decode(test_labels)}")
            else:
                logger.info(f"labels: {args.tokenizer.decode(labels[0].tolist())}")
            logger.info(f"cls_labels: {cls_labels[0].item()}")
            test_output = model.generate(input_ids=input_ids[0].unsqueeze(0))
            logger.info(f"outputs: {args.tokenizer.decode(test_output[0])}")
            # 如果是分类，打印分类结果
            if args.with_classifier:
                logger.info(f"classification_preds: {torch.argmax(classifier_output[0]).item()}")


    return total_loss / batch_id


# 评估函数
def evaluate(model, tokenizer, dataloader, logger, epoch, args):
    model.eval()
    rouge = Rouge()

    total_loss = 0
    total_char_f1 = 0
    num_samples = 0

    # 分类任务统计
    tp, fp, tn, fn = 0, 0, 0, 0
    total_classification_loss = 0

    predicted_texts = []
    reference_texts = []

    if args.with_classifier:
        logger.info("BART+classification")
    else:
        logger.info("BART")

    for batch in tqdm(dataloader, desc="Evaluating"):
        with torch.no_grad():

            input_ids = batch["input_ids"].to(args.device)
            labels = batch["labels"].to(args.device)
            cls_labels = batch["cls"].to(args.device)

            if args.with_classifier:
                # 分类任务评估
                encoder_outputs = model.model.encoder(input_ids)
                classifier_output = model.classifier(encoder_outputs.last_hidden_state[:, 0, :])
                predicted_cls_labels = torch.argmax(classifier_output, dim=1)
                classification_loss = nn.CrossEntropyLoss()(classifier_output, cls_labels).mean()
                total_classification_loss += classification_loss.item()

                # 将predicted_cls_labels中的2替换为1
                if args.with_three_classifier:
                    predicted_cls_labels[predicted_cls_labels == 2] = 1

                # 更新分类任务统计
                tp += ((predicted_cls_labels == 1) & (cls_labels == 1)).sum().item()
                fp += ((predicted_cls_labels == 1) & (cls_labels == 0)).sum().item()
                tn += ((predicted_cls_labels == 0) & (cls_labels == 0)).sum().item()
                fn += ((predicted_cls_labels == 0) & (cls_labels == 1)).sum().item()

                # 生成任务评估
                gen_mask = (predicted_cls_labels == 1)
                input_ids_gen = input_ids[gen_mask]

                if gen_mask.any():
                    outputs = model.generate(input_ids=input_ids_gen,
                                             max_length=args.generate_max_length, min_length=args.min_length,
                                             num_beams=args.num_beams, early_stopping=args.early_stopping,
                                             do_sample=args.do_sample, top_k=args.top_k, top_p=args.top_p,
                                             temperature=args.temperature, repetition_penalty=args.repetition_penalty,
                                             length_penalty=args.length_penalty,
                                             no_repeat_ngram_size=args.no_repeat_ngram_size)
                else:
                    outputs = []

                gen_outputs = model(input_ids=input_ids, labels=labels)

                output_idx = 0
                for idx in range(input_ids.size(0)):
                    tgt_text = tokenizer.decode(torch.where(labels[idx] == -100, tokenizer.pad_token_id, labels[idx]),
                                                skip_special_tokens=True)
                    reference_texts.append(tgt_text)

                    if gen_mask[idx]:
                        pred_text = tokenizer.decode(outputs[output_idx], skip_special_tokens=True)
                        output_idx += 1
                    else:
                        if args.is_woi:
                            pred_text = "No need to query"
                        else:
                            pred_text = "不需要查询"
                    predicted_texts.append(pred_text)

            else:
                outputs = model.generate(input_ids=input_ids,
                                         max_length=args.generate_max_length, min_length=args.min_length,
                                         num_beams=args.num_beams, early_stopping=args.early_stopping,
                                         do_sample=args.do_sample, top_k=args.top_k, top_p=args.top_p,
                                         temperature=args.temperature, repetition_penalty=args.repetition_penalty,
                                         length_penalty=args.length_penalty,
                                         no_repeat_ngram_size=args.no_repeat_ngram_size)
                gen_outputs = model(input_ids=input_ids, labels=labels)

                for idx in range(input_ids.size(0)):
                    tgt_text = tokenizer.decode(torch.where(labels[idx] == -100, tokenizer.pad_token_id, labels[idx]),
                                                skip_special_tokens=True)
                    reference_texts.append(tgt_text)

                    pred_text = tokenizer.decode(outputs[idx], skip_special_tokens=True)
                    if args.is_woi:
                        if pred_text == "":
                            pred_text = "No need to query"
                    else:
                        if pred_text == "":
                            pred_text = "不需要查询"
                    predicted_texts.append(pred_text)

            # 计算评估指标时，将“不需要查询”替换为“不”
            if args.is_woi:
                reference_texts_char_f1 = [text.replace("No need to query", "N") for text in reference_texts]
                predicted_texts_char_f1 = [text.replace("No need to query", "N") for text in predicted_texts]
            else:
                reference_texts_char_f1 = [text.replace("不需要查询", "不") for text in reference_texts]
                predicted_texts_char_f1 = [text.replace("不需要查询", "不") for text in predicted_texts]

            batch_char_f1 = 0  # 初始化当前批次的总Char-F1分数
            for a, b in zip(predicted_texts_char_f1, reference_texts_char_f1):
                # 捕捉这里的异常
                try:
                    rouge_scores = rouge.get_scores(a, b)
                except:
                    logger.info("Exception: a={}, b={}".format(a, b))
                char_f1 = rouge_scores[0]['rouge-1']['f']
                batch_char_f1 += char_f1  # 更新当前批次的总Char-F1分数

            total_char_f1 += batch_char_f1  # 更新总Char-F1分数

            if gen_outputs is not None:
                generation_loss = gen_outputs.loss.mean().item()
                total_loss += generation_loss

            num_samples += 1

    # 计算评估指标
    if args.with_classifier:
        # 修正精准率、召回率、F1值分母为0的情况
        classification_accuracy = (tp + tn) / (tp + tn + fp + fn)  # 修改这里
        if tp + fp == 0:
            classification_precision = 0
        else:
            classification_precision = tp / (tp + fp)
        if tp + fn == 0:
            classification_recall = 0
        else:
            classification_recall = tp / (tp + fn)
        if classification_precision + classification_recall == 0:
            classification_f1 = 0
        else:
            classification_f1 = 2 * classification_precision * classification_recall / (
                    classification_precision + classification_recall)
        avg_classification_loss = total_classification_loss / num_samples

    if args.is_woi:
        num_not_required = sum([1 for text in predicted_texts if text == "No need to query"])
    else:
        num_not_required = sum([1 for text in predicted_texts if ''.join(text.split()) == "不需要查询"])
    not_required_ratio = (num_not_required / len(predicted_texts)) * 100

    avg_generation_loss = total_loss / num_samples
    avg_char_f1 = total_char_f1 / num_samples

    if args.with_classifier:
        all_loss = avg_generation_loss + avg_classification_loss
    else:
        all_loss = avg_generation_loss

    # 输出评估结果
    logger.info("epoch: {}".format(epoch))
    logger.info("Evaluation Results:")
    if args.with_classifier:
        logger.info(f"Classification Loss: {avg_classification_loss}")
        logger.info("Classification Accuracy: {:.2f}%".format(classification_accuracy * 100))
        logger.info("Classification Precision: {:.2f}%".format(classification_precision * 100))
        logger.info("Classification Recall: {:.2f}%".format(classification_recall * 100))
        logger.info("Classification F1: {:.2f}".format(classification_f1 * 100))
    logger.info(f"Generation Loss: {avg_generation_loss}")
    logger.info("Char-F1: {:.2f}%".format(avg_char_f1 * 100))
    # 输出 '不需要查询' 的比例
    logger.info("不需要查询的比例: {:.2f}%".format(not_required_ratio))

    if args.infer:
        with open(args.infer_result_path, 'w', encoding='utf-8') as f:
            logger.info(f"推理结果保存到：{args.infer_result_path}")
            for i, pred_text in enumerate(predicted_texts):
                # print(f"reference: {reference_texts[i]}")
                # print(f"predicted: {pred_text}")
                f.write(pred_text + '\n')

    return all_loss


# 主函数
def main(args):
    set_seed(args.seed)

    logger = set_logger(args.log_file)
    logger.info(f"Args: {json.dumps(vars(args), indent=2)}")

    if args.is_woi:
        tokenizer = BartTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = BertTokenizer.from_pretrained(args.model_name)
    args.tokenizer = tokenizer

    if args.infer and args.infer_model_path:
        logger.info(f"推理模式，从加载模型：{args.infer_model_path}")
        model = BartForConditionalGeneration.from_pretrained(args.infer_model_path).to(args.device)
    else:
        logger.info(f"训练模式，加载模型：{args.model_name}")
        model = BartForConditionalGeneration.from_pretrained(args.model_name).to(args.device)

    num_classes = 3 if args.with_three_classifier else 2
    if args.with_classifier:
        logger.info(f"定义分类头：{args.model_name}")
        model.classifier = nn.Sequential(
            nn.Linear(model.config.d_model, int(model.config.d_model / 2)),
            nn.ReLU(),
            nn.Linear(int(model.config.d_model / 2), num_classes)
        ).to(args.device)

    if args.infer and args.infer_model_path and args.with_classifier:
        logger.info(f"推理模式，加载分类头：{args.infer_model_path + '/classifier.pt'}")
        model.classifier.load_state_dict(torch.load(args.infer_model_path + "/classifier.pt"))

    train_data, valid_data, test_data = load_data(args, logger)
    if args.infer:
        logger.info(f"推理模式，对test进行测试")
        valid_data = test_data

    train_dataset = Seq2SeqAndClassificationDataset(tokenizer, train_data["src"], train_data["tgt"], train_data["cls"],
                                                    train_data["cls_3"], args)
    valid_dataset = Seq2SeqAndClassificationDataset(tokenizer, valid_data["src"], valid_data["tgt"], valid_data["cls"],
                                                    valid_data["cls_3"], args)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  collate_fn=lambda batch: seq2seq_and_classification_collate_fn(batch, tokenizer, args),
                                  pin_memory=args.pin_memory)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=args.pin_memory,
                                  collate_fn=lambda batch: seq2seq_and_classification_collate_fn(batch, tokenizer, args))

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=args.warmup_steps,
                                                             num_training_steps=len(train_dataloader) * args.epochs)
    if not args.infer:
        train_loss_list = []
        eval_loss_list = []
        for epoch in range(args.epochs):
            logger.info(f"==================== Epoch: {epoch + 1} ====================")
            train_loss = train(model, train_dataloader, optimizer, scheduler, logger, args)
            logger.info(f"Train loss: {train_loss}")
            train_loss_list.append(train_loss)

            eval_loss = evaluate(model, tokenizer, valid_dataloader, logger, epoch + 1, args)
            logger.info(f"Eval loss: {eval_loss}")
            eval_loss_list.append(eval_loss)

            logger.info(f"train_loss_list: {train_loss_list}")
            logger.info(f"eval_loss_list: {eval_loss_list}")

            # 判断路径是否存在，不存在则创建
            if not os.path.exists(os.path.join(args.save_path, f'epoch_{epoch + 1}')):
                os.makedirs(os.path.join(args.save_path, f'epoch_{epoch + 1}'))
            logger.info(f"Saving model to {os.path.join(args.save_path, f'epoch_{epoch + 1}')}")
            model.save_pretrained(os.path.join(args.save_path, f"epoch_{epoch + 1}"))
            if args.with_classifier:
                logger.info(
                    f"Saving classifier to {os.path.join(args.save_path, f'epoch_{epoch + 1}', 'classifier.pt')}")
                torch.save(model.classifier.state_dict(),
                           os.path.join(args.save_path, f"epoch_{epoch + 1}", "classifier.pt"))
        logger.info("Training completed.")
    else:
        logger.info("直接推理，不进行训练")
        evaluate(model, tokenizer, valid_dataloader, logger, 0, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 文件
    parser.add_argument("--data_path", type=str, default='./DuSinc.json',
                        help="Path to the data files.")
    parser.add_argument("--model_name", type=str, default="fnlp/bart-base-chinese", help="Model name or path.")
    parser.add_argument("--log_file", type=str, default="log/train_duinc.log", help="Path to the log file.")
    parser.add_argument("--save_path", type=str, default="./model/dusinc", help="Path to save the trained model.")
    parser.add_argument("--is_woi", action='store_true', help="是否为woi")
    # 训练方法
    parser.add_argument("--with_classifier", action='store_false', help="加入分类")
    parser.add_argument("--with_three_classifier", action='store_false', help="加入三分类")
    parser.add_argument("--with_keywords", action='store_false', help="加入关键词")

    # 超参数
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--warmup_steps", type=int, default=1500, help="Warmup up step")
    parser.add_argument("--seed", type=int, default=2023, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", action='store_false')
    parser.add_argument("--log_steps", type=int, default=5000, help="Log steps / batch.")
    parser.add_argument("--classification_weight", type=float, default=5.0, help="分类任务权重")
    parser.add_argument("--generation_weight", type=float, default=1.0, help="生成任务权重")
    parser.add_argument("--not_use_amp", action='store_true', help="混合精度")
    # 推理参数
    parser.add_argument("--infer", action='store_true', help="推理模式")
    parser.add_argument("--infer_model_path", type=str, default="./model/dusinc")
    parser.add_argument("--infer_result_path", type=str, default="./dusinc_CQP_predictions.txt")
    # 生成参数
    parser.add_argument("--num_beams", type=int, default=10, help="Number of beams for beam search.")
    parser.add_argument("--length_penalty", type=float, default=1.0, help="Length penalty for generation.")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3, help="No repeat ngram size for generation.")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length for generation.")
    parser.add_argument("--generate_max_length", type=int, default=30, help="Maximum length for generation.")
    parser.add_argument("--early_stopping", action='store_true', help="Early stopping for generation.")
    parser.add_argument("--do_sample", action='store_false', help="Do sample for generation.")
    parser.add_argument("--top_k", type=int, default=0, help="Top k for generation.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top p for generation.")
    parser.add_argument("--temperature", type=float, default=0.95, help="Temperature for generation.")
    parser.add_argument('--repetition_penalty', default=1.0, type=float, required=False, help='重复惩罚参数')

    args = parser.parse_args()

    # 修改保存路径
    import time
    import platform
    args.save_path = os.path.join(args.save_path, f"{time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())}_{platform.node()}")

    main(args)
