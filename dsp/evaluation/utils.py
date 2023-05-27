import dsp
import tqdm
import pandas as pd

from IPython.display import display
from dsp.utils import EM, F1, HotPotF1


def evaluateRetrieval(fn, dev, metric=None):
    data = []

    for example in tqdm.tqdm(dev):
        question = example.question
        prediction = fn(question)

        d = dict(example)

        # d['prediction'] = prediction.answer
        d['correct'] =  dsp.passage_match(prediction.context, example.answer)
        data.append(d)

    df = pd.DataFrame(data)

    percentage = round(100.0 * df['correct'].sum() / len(dev), 1)
    print(f"Answered {df['correct'].sum()} / {len(dev)} ({percentage}%) correctly.")
    df['correct'] = df['correct'].apply(lambda x: '✔️' if x else '❌')

    pd.options.display.max_colwidth = None
    display(df.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}, {'selector': 'td', 'props': [('text-align', 'left')]}]))


def evaluateAnswer(fn, dev, metric=EM):
    data = []

    for example in tqdm.tqdm(dev):
        question = example.question
        prediction = fn(question)

        d = dict(example)

        pred = prediction.answer

        d['prediction'] = pred
        d['correct'] = metric(pred, example.answer)
        data.append(d)

    df = pd.DataFrame(data)

    percentage = round(100.0 * df['correct'].sum() / len(dev), 1)
    print(f"Answered {df['correct'].sum()} / {len(dev)} ({percentage}%) correctly.")
    df['correct'] = df['correct'].apply(lambda x: '✔️' if x else '❌')

    pd.options.display.max_colwidth = None
    display(df.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}, {'selector': 'td', 'props': [('text-align', 'left')]}]))

def evaluate(fn, dev, metric=EM, return_df=False):
    data = []

    for example in tqdm.tqdm(dev):
        question = example.question
        try:
            prediction = fn(question)
        except:
            #print(f"An error happened processing Question: {question} \n\n")
            prediction = "Error"

        d = dict(example)

        pred = prediction#.answer

        d['prediction'] = pred
        if d['prediction'] != "Error":
            d['correct'] = metric(pred, example.answer)
        else:
            d['correct'] = False
        data.append(d)

    df = pd.DataFrame(data)

    percentage = round(100.0 * df['correct'].sum() / len(dev), 1)
    print(f"Answered {df['correct'].sum()} / {len(dev)} ({percentage}%) correctly.")
    df['correct'] = df['correct'].apply(lambda x:'⚠️' if x == 'Error' else '✔️' if x else '❌')

    pd.options.display.max_colwidth = None
    display(df.style.set_table_styles([{'selector': 'th', 'props': [('text-align', 'left')]}, {'selector': 'td', 'props': [('text-align', 'left')]}]))
    if not return_df:
        return percentage
    else:
        return percentage, df