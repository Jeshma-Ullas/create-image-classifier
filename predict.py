import argparse
import numpy as np
import json
import fmodel

def parse_arguments():
    parser = argparse.ArgumentParser(description='Parser for predict.py')
    parser.add_argument('input', nargs='?', default='./flowers/test/1/image_06752.jpg', action="store", type=str)
    parser.add_argument('--dir', action="store", dest="data_dir", default="./flowers/")
    parser.add_argument('checkpoint', nargs='?', default='./checkpoint.pth', action="store", type=str)
    parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
    parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
    parser.add_argument('--gpu', default=False, action="store_true", dest="gpu")

    return parser.parse_args()

def predict(image_path, checkpoint_path, category_names, topk, device):
    model = fmodel.load_checkpoint(checkpoint_path)
    with open(category_names, 'r') as json_file:
        name = json.load(json_file)
        
    probabilities = fmodel.predict(image_path, model, topk, device)
    probability = np.array(probabilities[0][0])
    labels = [name[str(index + 1)] for index in np.array(probabilities[1][0])]
    
    for i in range(topk):
        print("{} with a probability of {}".format(labels[i], probability[i]))
    print("Finished Predicting!")

def main():
    args = parse_arguments()
    path_image = args.input
    number_of_outputs = args.top_k
    device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'
    json_name = args.category_names
    path = args.checkpoint
    
    predict(path_image, path, json_name, number_of_outputs, device)
    
if __name__ == "__main__":
    main()
