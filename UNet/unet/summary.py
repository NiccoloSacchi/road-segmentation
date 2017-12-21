
import os
import os.path
from collections import defaultdict

import numpy as np
import h5py

class Summary(object):

    def __init__(self):

        self.content = defaultdict(dict)

    def register(self, tag, index, value):

        self.content[tag][index] = value

    def get(self, tag):

        if tag not in self.content:
            raise KeyError(tag)

        data = self.content[tag]

        indices = []
        values = []
        for index in sorted(data):
            indices.append(index)
            values.append(data[index])

        return np.asarray(indices), np.asarray(values)

    def save(self, filename, backup=False):

        if backup and os.path.isfile(filename):
            os.rename(filename, filename + ".bak")

        np.save(filename, self.content)

    def load(self, filename):
        self.content = np.load(filename).item()


def save_h5(summary, filename):

    f = h5py.File(filename, "w")
    for tag in summary.content.keys():
        grp = f.create_group(tag)

        indices, values = summary.get(tag)

        grp.create_dataset("indices", data=indices)
        grp.create_dataset("values", data=values)

    f.close()

def load_h5(summary, filename=None):

    if filename is None:
        filename = summary
        summary = Summary()

    f = h5py.File(filename, "r")

    content = defaultdict(dict)

    for tag in f.keys():
        grp = f[tag]
        indices = grp["indices"][:]
        values = grp["values"][:]

        for index, value in zip(indices, values):
            content[tag][index] = value

    f.close()
    summary.content = content

    return summary
