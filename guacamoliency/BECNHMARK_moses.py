import moses
import pandas as pd
import argparse
import numpy as np

def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input_dir', type = str, default="data/generated/moses_canonical_CL_1.csv",
                        help="where is your model file", required=True)

    parser.add_argument('--output_dir', type = str, default='reports/data/BENCHMARK.csv',
                        help="where save our outputs", required=True)
    
    parser.add_argument('--model_name', type = str, default='MolGPT_moses_canonical_characterlevel_1',
                        help="name of the type of model used", required=True)
    
    parser.add_argument('--number_worker', type = int, default=1,
                        help="number of worker", required=False)

    args = parser.parse_args()

    df = pd.read_csv(args.input_dir)
    dataset = df["SMILES"].to_list()

    dataset = [k for k in dataset if isinstance(k,str)]
    data_length = len(dataset)

    metrics = moses.get_all_metrics(dataset,n_jobs = args.number_worker)
    print(metrics)
    metrics["model_name"] = args.model_name
    metrics["sample length"] = data_length

    try :
        benchmark_file = pd.read_csv(args.output_dir)
        benchmark_file=pd.concat([benchmark_file,pd.DataFrame.from_dict([metrics])],ignore_index=True)
    except Exception as e:
        print(e)
        benchmark_file = pd.DataFrame.from_dict([metrics])
    
    
    benchmark_file.to_csv(args.output_dir,index=False)

if __name__ == "__main__":
    main()