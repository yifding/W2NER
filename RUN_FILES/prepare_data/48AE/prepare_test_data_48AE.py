import os
import argparse
import jsonlines
from functools import partial
from tqdm.contrib.concurrent import process_map
from tqdm import tqdm


def process_instance(instance, max_span_length=5, max_seq_length=300):
    asin = instance.get('asin', '')
    product_type = instance.get('product_type', '')
    tokens = instance["X_text"]
    tokens_lower = [token.lower() for token in tokens]
    tokens_lower = tokens_lower[:max_seq_length]

    gt_spans = []
    att = list(instance['Y_values'].keys())[0]
    attribute_values = instance['Y_values'][att]
    for i in range(len(tokens_lower) - 1):
        for j in range(i + 1, min(len(tokens_lower)+1, i+1+max_span_length)):
            select_tokens = tokens_lower[i:j]
            pred_text = ' '.join(select_tokens)

            if pred_text not in attribute_values:
                continue

            gt_span = {
                "index": list(range(i, j)),
                "type": att,
            }
            gt_spans.append(gt_span)

    gt_spans = sorted(gt_spans, key=lambda x: (x["index"][0], len(x["index"])))
    re_instance = {
        "asin": asin,
        "product_type": product_type,
        "ner": gt_spans,
        "sentence": tokens_lower,
    }
    return re_instance


def prepare_test_data(
    input_file,
    output_file,
    max_span_length,
    max_seq_length,
):
    instances = []
    with jsonlines.open(input_file, 'r') as reader:
        for instance in reader:
            instances.append(instance)

    # re_instances = process_map(
    #     partial(process_instance, att_list=att_list, max_span_length=max_span_length),
    #     instances,
    #     chunksize=1,
    #     max_workers=8,
    # )
    re_instances = []
    for instance in tqdm(instances):
        re_instance = process_instance(instance, max_span_length=max_span_length, max_seq_length=max_seq_length)
        re_instances.append(re_instance)

    with jsonlines.open(output_file, 'w') as writer:
        writer.write_all(re_instances)


def parse_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--input_dir",
        required=True,
        # default="/yifad_ebs/consumable/clean_test_data/Color.gold"
        type=str,
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        # default="/yifad_ebs/AVEQA_PyTorch/RUN_FILES/08_15_2021/qa_15att/train_qa_neg_1",
        type=str,
    )
    parser.add_argument(
        "--max_span_length",
        default=5,
        help="maximum length (number of words) of span",
        type=int,
    )
    parser.add_argument(
        "--max_seq_length",
        default=300,
        help="maximum length (number of words) of sequence",
        type=int,
    )
    parser.add_argument(
        "--att_list",
        required=True,
        # default="['ActiveIngredients','AgeRangeDescription','BatteryCellComposition','Brand','CaffeineContent','CapacityUnit','CoffeeRoastType','Color','DietType','DosageForm','EnergyUnit','FinishType','Flavor','FormulationType','HairType','Ingredients','ItemForm','ItemShape','LiquidContentsDescription','Material','MaterialFeature','MaterialTypeFree','PackageSizeName','Pattern','PatternType','ProductBenefit','Scent','SkinTone','SkinType','SpecialIngredients','TargetGender','TeaVariety','Variety']",
        type=eval,
    )

    args = parser.parse_args()
    assert os.path.isdir(args.input_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    return args


def main():
    args = parse_args()
    for att in args.att_list:
        input_file = os.path.join(args.input_dir, f'{att}.gold')
        output_file = os.path.join(args.output_dir, f'{att}.json')
        prepare_test_data(
            input_file=input_file,
            output_file=output_file,
            max_span_length=args.max_span_length,
            max_seq_length=args.max_seq_length,
        )


if __name__ == "__main__":
    main()