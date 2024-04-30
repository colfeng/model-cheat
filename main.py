from data import data_pre
from model import model_pre
from infer import main_infer
import argparse

import json
def res_json_save(ppl, save_path):
    ppl_value = {"train_ppl":ppl[0], "test_ppl":ppl[1], "DLT":ppl[2]}
    save_dic = json.dumps(ppl_value)
    f = open(save_path, 'w')
    f.write(save_dic)
    f.close()


def main(model_path,
         train_data_path,
         test_data_path,
         save_path,
         device = "cuda",
         device_ids = [0],):
    train_data =  data_pre(train_data_path)
    test_data = data_pre(test_data_path)
    model, tokenizer = model_pre(model_path)
    train_ppl = main_infer(train_data, model, tokenizer, device, device_ids, save_path)
    test_ppl = main_infer(test_data, model, tokenizer, device, device_ids, save_path)
    DLT = test_ppl - train_ppl
    res_json_save([test_ppl, train_ppl, DLT], save_path)

    print("The PPL of this model in test datasets is", test_ppl)
    print("The PPL of this model in train datasets is", train_ppl)
    print("The DLT of this model in this datasets is", DLT)
    print("saved and finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # experiment params
    parser.add_argument('--model_path', default="/data/fengduanyu/llm/llama2/7B/", help="model path")
    parser.add_argument('--train_data_path', default="/data/fengduanyu/modelcheat/test_data/test.jsonl",
                        help='train datasets path')
    parser.add_argument('--test_data_path', default="/data/fengduanyu/modelcheat/test_data/test.jsonl",
                        help='test datasets path')
    parser.add_argument('--save_path', default='/data/fengduanyu/modelcheat/res/llama2-7B.json', help='result save path')
    parser.add_argument('--device', default='cuda', help='Device')
    parser.add_argument("--device_ids", type=list, default=[0])#[0,1,2,3,4,5,6,7]
    args = parser.parse_args()

    main(model_path = args.model_path,
         train_data_path = args.train_data_path,
         test_data_path = args.test_data_path,
         save_path=args.save_path,
         device = args.device,
         device_ids = args.device_ids,
         )