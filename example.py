import os
from pathlib import Path
from nll import NLLCal
from autoperiod import AutoPeriod
import numpy as np


def text_to_nll(text_path, model_path):
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read().strip()
    texts = [text]

    # nll_cal = NLLCal(model_path=model_path)
    nll_cal = NLLCal(
        model_path=model_path,
        max_length=1024,
        batch_size=4,
    )
    nll = nll_cal.texts_to_nll(texts)[0].float().cpu().numpy()
    return nll


def main():

    model_path = "/data1/model/gpt2-small"

    nll = text_to_nll('test/t1.txt', model_path)
    # np.save('test/t1.npy', nll)

    # nll = np.load('test/t1.npy') ##

    times = np.arange(len(nll))
    # run autoperoid
    results = AutoPeriod(
        times,
        nll,
        mc_iterations=1000,
        confidence_level=.99,
    ).run()
    print(results)


if __name__ == "__main__":
    main()
