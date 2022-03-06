import os
import argparse
import json
import jsonlines

import utils
import config
from data_loader import RelationDataset, process_bert, Vocabulary, collate_fn
from model import Model

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


def load_data_bert_test(test_file, config):
    with jsonlines.open(test_file, 'r') as f:
        test_data = [line for line in f]

    tokenizer = AutoTokenizer.from_pretrained(config.bert_name, cache_dir="./cache/")
    test_dataset = RelationDataset(*process_bert(test_data, tokenizer, config.vocab))
    return test_dataset, test_data


def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text


def decode(outputs, entities, length):
    ent_r, ent_p, ent_c = 0, 0, 0
    predicts = set()
    for index, (instance, ent_set, l) in enumerate(zip(outputs, entities, length)):
        forward_dict = {}
        head_dict = {}
        ht_type_dict = {}
        for i in range(l):
            for j in range(i + 1, l):
                if instance[i, j] == 1:
                    if i not in forward_dict:
                        forward_dict[i] = [j]
                    else:
                        forward_dict[i].append(j)
        for i in range(l):
            for j in range(i, l):
                if instance[j, i] > 1:
                    ht_type_dict[(i, j)] = instance[j, i]
                    if i not in head_dict:
                        head_dict[i] = {j}
                    else:
                        head_dict[i].add(j)

        predicts = []

        def find_entity(key, entity, tails):
            entity.append(key)
            if key not in forward_dict:
                if key in tails:
                    predicts.append(entity.copy())
                entity.pop()
                return
            else:
                if key in tails:
                    predicts.append(entity.copy())
            for k in forward_dict[key]:
                find_entity(k, entity, tails)
            entity.pop()

        for head in head_dict:
            find_entity(head, [], head_dict[head])

        predicts = set([convert_index_to_text(x, ht_type_dict[(x[0], x[-1])]) for x in predicts])

        ent_r += len(ent_set)
        ent_p += len(predicts)
        for x in predicts:
            if x in ent_set:
                ent_c += 1
    return predicts


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/conll03.json')
    parser.add_argument('--device', type=int, default=0)

    parser.add_argument('--dist_emb_size', type=int)
    parser.add_argument('--type_emb_size', type=int)
    parser.add_argument('--lstm_hid_size', type=int)
    parser.add_argument('--conv_hid_size', type=int)
    parser.add_argument('--bert_hid_size', type=int)
    parser.add_argument('--ffnn_hid_size', type=int)
    parser.add_argument('--biaffine_size', type=int)

    parser.add_argument('--dilation', type=str, help="e.g. 1,2,3")

    parser.add_argument('--emb_dropout', type=float)
    parser.add_argument('--conv_dropout', type=float)
    parser.add_argument('--out_dropout', type=float)

    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)

    parser.add_argument('--clip_grad_norm', type=float)
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--weight_decay', type=float)

    parser.add_argument('--bert_name', type=str)
    parser.add_argument('--bert_learning_rate', type=float)
    parser.add_argument('--warm_factor', type=float)

    parser.add_argument('--use_bert_last_4_layers', type=int, help="1: true, 0: false")

    parser.add_argument('--seed', type=int)

    # **YD** add model_dir and dataset_dir to store/load model and load dataset
    parser.add_argument('--pretrained_checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--att_list', type=eval, required=True, help='number of attributes from the model')
    parser.add_argument('--test_att_list', type=eval, required=True, help='number of attributes from the test files')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    return args


if __name__ == '__main__':

    args = parse_args()
    config = config.Config(args)

    logger = utils.get_logger(config.dataset)
    logger.info(config)
    config.logger = logger

    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)

    # random.seed(config.seed)
    # np.random.seed(config.seed)
    # torch.manual_seed(config.seed)
    # torch.cuda.manual_seed(config.seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    vocab = Vocabulary()
    for att in args.att_list:
        vocab.add_label(att)
    print(len(vocab.label2id))
    print(vocab.label2id)
    config.label_num = len(vocab.label2id)
    config.vocab = vocab

    model = Model(config)

    print("yd: start to load ckpt !")
    model.load_state_dict(
        torch.load(args.pretrained_checkpoint,
                   map_location=torch.device('cpu'))
    )
    print("yd: finish loading ckpt !")

    model = model.cuda()
    model.eval()

    for test_att in args.test_att_list:
        test_file = os.path.join(args.test_dir, f'{test_att}.json')
        test_dataset, test_data = load_data_bert_test(test_file, config)

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=1,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=1,
        )

        test_att_index = vocab.label2id[test_att.lower()]
        output_file = os.path.join(args.output_dir, f'{test_att}.json')
        output_dict = dict()
        for i, data_batch in enumerate(test_loader):
            test_instance = test_data[i]
            entity_text = data_batch[-1]
            data_batch = [data.cuda() for data in data_batch[:-1]]
            bert_inputs, grid_labels, grid_mask2d, pieces2word, dist_inputs, sent_length = data_batch

            outputs = model(bert_inputs, grid_mask2d, dist_inputs, pieces2word, sent_length)
            length = sent_length

            grid_mask2d = grid_mask2d.clone()

            outputs = torch.argmax(outputs, -1)
            predicts = decode(outputs.cpu().numpy(), entity_text, length.cpu().numpy())

            # print(f"test_instance: {test_instance['ner']}; {test_instance['sentence']}")
            # print(f"entity_text: {entity_text[0]}")
            # print(f"predicts: {predicts}")

            output_prediction_list = []
            """
            entity_text: {'84-#-2', '90-#-2', '41-#-2', '78-79-#-2', '3-#-2', '84-85-#-2', '90-91-#-2'}
            predicts: {'45-#-11'}
            """

            for predict in predicts:
                tmp_predict = []
                def process_entity(entity_str):
                    parts = entity_str.split('-')
                    assert parts.count('#') == 1
                    star_index = parts.index('#')
                    assert parts[star_index+1] == parts[-1]
                    token_index_list = [int(num) for num in parts[:star_index]]
                    att_index = int(parts[star_index+1])
                    return att_index, token_index_list

                tmp_att_index, tmp_token_index_list = process_entity(predict)
                if tmp_att_index == test_att_index:
                    for tmp_token_index in tmp_token_index_list:
                        tmp_predict.append(test_instance['sentence'][tmp_token_index])
                output_prediction_list.append(' '.join(tmp_predict))

            output_dict[i] = output_prediction_list

        with open(output_file, 'w') as writer:
            json.dump(output_dict, writer, indent=4)

