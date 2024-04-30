from data import data_pre
from model import model_pre
from infer import main_infer
import argparse

def main(model_path,
         dataset_path,
         save_path,
         device = "cuda",
         device_ids = [0],):
    data =  data_pre(dataset_path)
    model, tokenizer = model_pre(model_path)
    main_infer(data, model, tokenizer, device, device_ids, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # experiment params
    parser.add_argument('--model_path', default="/data/fengduanyu/llm/llama2/7B/", help="model path")
    parser.add_argument('--dataset_path', default="/data/fengduanyu/modelcheat/test_data/test.jsonl", help='datasets path')  # 'alpaca' InstructionWild
    parser.add_argument('--save_path', default='/data/fengduanyu/modelcheat/res/llama2-7B.json', help='result save path')
    parser.add_argument('--device', default='cuda', help='Device')
    parser.add_argument("--device_ids", type=list, default=[0])#[0,1,2,3,4,5,6,7]
    args = parser.parse_args()

    main(model_path = args.model_path,
         dataset_path = args.dataset_path,
         save_path=args.save_path,
         device = args.device,
         device_ids = args.device_ids,
         )