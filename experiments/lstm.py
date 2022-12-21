
import pandas as pd
import argparse
import re


from offensive_nn.offensive_nn_model import OffensiveNNModel
from offensive_nn.util.print_stat import print_information
from sklearn.model_selection import train_test_split
from offensive_nn.util.label_converter import encode, decode
import numpy as np
from experiments.config import args

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default=None)
parser.add_argument('--algorithm', required=False, help='algorithm', default="lstm")  # lstm or cnn2D
parser.add_argument('--train', required=False, help='train file', default='../data/prepocess.csv')
arguments = parser.parse_args()

df = pd.read_csv(arguments.train, sep=",", index_col=0)
df.dropna(subset=['testimony'], inplace=True)
df.dropna(subset=['A1'], inplace=True)

li = []
# regext after the content text
for i in df['testimony']:
    text = re.search(r'(Contents[^*]+)', i)
    if text:
        r = text.group(1)
        li.append(r)
    else:
        r = None
        li.append(r)
        # print(text.group(1))
df['new'] = li
df.dropna(subset=['new'], inplace=True)
# get the count of words in individual testimony
df['doc_len'] = df['new'].apply(lambda words: len(words.split(" ")))
max_seq_len = np.round(df['doc_len'].mean() + df['doc_len'].std()).astype(int)

# split data into train and test label
# x_train, x_test= train_test_split(df,test_size=0.2, random_state=434)


df = df.rename(columns={'new': 'text','A1': 'labels'})

test_sentences = df['text'].tolist()

test_preds = np.zeros((len(df), args["n_fold"]))




train_set = df
test_set = df




test_set['labels'] = encode(test_set['labels'])
for i in range(args["n_fold"]):
    train_set, validation_set = train_test_split(train_set, test_size=0.2, random_state=args["manual_seed"])
    model = OffensiveNNModel(model_type_or_path=arguments.algorithm, embedding_model_name=arguments.model_name,
                             train_df=train_set,
                             args=args, eval_df=validation_set)
    model.train_model()
    print("Finished Training")
    model = OffensiveNNModel(model_type_or_path=args["best_model_dir"])
    predictions, raw_outputs = model.predict(test_sentences)
    test_preds[:, i] = predictions
    print("Completed Fold {}".format(i))

final_predictions = []
for row in test_preds:
    row = row.tolist()
    final_predictions.append(int(max(set(row), key=row.count)))

print(final_predictions)

test_set['predictions'] = final_predictions

print_information(test_set, "predictions", "labels")