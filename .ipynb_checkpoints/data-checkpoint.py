import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split


class TrajDataset(Dataset):
    def __init__(self, file_name, seq_lens=100):
        self.file_name = file_name
        self.seq_lens = seq_lens
        self.Fs_p, self.Fs_f, self.L_p, self.L_f, self.Lab_p, self.Lab_f = self._create_samples()


    def __len__(self):
        return len(self.Fs_f)


    def __getitem__(self, index):
        return self.Fs_p[index], self.Fs_f[index], self.L_p[index], self.L_f[index], self.Lab_p[index], self.Lab_f[index]
    def _create_samples(self):
        data = pd.read_csv(self.file_name)
        # labels process
        Lab = data["label"].values

        for (i, lab) in enumerate(Lab):
            if lab.endswith("sailing"):
                Lab[i] = 0
            else:
                Lab[i] = 1

        data["label"] = Lab

        ids = data["id"].unique()
        L = [len(data[data["id"] == iid]) for iid in ids]

        # data pre-processing
        D = []
        for iid, l in zip(ids, L):
            
            d = data.loc[
                data["id"] == iid, ["signed_turn", "distance_gap", "euc_speed", "distanceToShore", "time_gap", "label"]]
            d.reset_index(drop=True, inplace=True)
            
            inds = np.argwhere(d["time_gap"].values > 600)
            inds = [ii[0] for ii in inds]
            if len(inds) > 0:

                D.append(d.loc[0:inds[0] - 1,
                         ["signed_turn", "distance_gap", "euc_speed", "distanceToShore", "time_gap", "label"]])
                for i in range(len(inds) - 1):
                    D.append(d.loc[inds[i]:inds[i] - 1,
                             ["signed_turn", "distance_gap", "euc_speed", "distanceToShore", "time_gap", "label"]])
                D.append(d.loc[inds[-1]:,
                         ["signed_turn", "distance_gap", "euc_speed", "distanceToShore", "time_gap", "label"]])
            else:
                D.append(d.loc[:, ["signed_turn", "distance_gap", "euc_speed", "distanceToShore", "time_gap", "label"]])
        D = [d for d in D if len(d) > 1000]
        for d in D:
            d.reset_index(drop=True, inplace=True)
        L = [len(d) for d in D]

        for d in D:
            d.loc[d["euc_speed"] > 20, "euc_speed"] = 20
            d["euc_speed"] = d["euc_speed"] * 0.5144
            d.loc[d["distance_gap"] > 200, "distance_gap"] = 200
            d["signed_turn"] = d["signed_turn"] / 180 * np.pi

        Fs = []
        Lab = []
        # feature calculation
        for d in D:

            speed = d["euc_speed"].values[1:]
            signed_turn = d["signed_turn"].values[1:]
            distance_gap = d["distance_gap"].values[1:]
            distanceToShore = d["distanceToShore"].values

            time_gap = d["time_gap"].values[1:]

            speed_gap = d["euc_speed"].diff().values[1:]
            speed_gap_ratio = speed_gap / time_gap
            distance_gap_ratio = distance_gap / time_gap
            signed_turn_ratio = signed_turn / time_gap
            distanceToShore_gap = np.diff(distanceToShore) / time_gap

            Fs.append(np.array(
                [speed.tolist(), signed_turn.tolist(), distance_gap.tolist(), distanceToShore[1:].tolist(),
                 speed_gap.tolist(),
                 speed_gap_ratio.tolist(), distance_gap_ratio.tolist(), signed_turn_ratio.tolist(),
                 distanceToShore_gap.tolist()]).transpose(1, 0))
            Lab.append(d["label"].values[1:])

        L = [len(f) for f in Fs]
        V = np.concatenate(Fs, axis=0)

        Fs = [(fs - np.mean(V, axis=0)) / (np.std(V, axis=0)) for fs in Fs]

        Fs_p, L_p, Fs_f, L_f, Flag, P,Lab_p, Lab_f = self.sample(Fs, Lab, self.seq_lens)

        Fs_p = np.stack(Fs_p, axis=0)
        Fs_f = np.stack(Fs_f, axis=0)
        L_p = np.stack(L_p, axis=0)
        L_f = np.stack(L_f, axis=0)

        Fs_p = torch.tensor(Fs_p)
        Fs_f = torch.tensor(Fs_f)
        L_p = torch.tensor(L_p.astype(np.int32))
        L_f = torch.tensor(L_f.astype(np.int32))

        Fs_p = Fs_p.float()
        Fs_f = Fs_f.float()

        return Fs_p, Fs_f, L_p, L_f, Lab_p, Lab_f

    def sample(self, feature, lab, window_size):
        Flag = []
        Fs_p = []
        Fs_f = []
        L_p = []
        L_f = []
        P = []
        Lab_p = []
        Lab_f = []
        for i in range(len(feature)):
            f = feature[i]
            l = lab[i]
            p = []
            for j in range(window_size, l.shape[0] - window_size, window_size):
                Fs_p.append(f[j - window_size:j, :])
                L_p.append(l[j - window_size:j])
                if len(np.unique(l[j - window_size:j])) == 2:
                    Lab_p.append(2)
                else:
                    if np.unique(l[j - window_size:j]) == 0:
                        Lab_p.append(0)
                    else:
                        Lab_p.append(1)

                Fs_f.append(f[j:j + window_size, :])
                L_f.append(l[j:j + window_size])
                if len(np.unique(l[j:j + window_size])) == 2:
                    Lab_f.append(2)
                else:
                    if np.unique(l[j:j + window_size]) == 0:
                        Lab_f.append(0)
                    else:
                        Lab_f.append(1)
                Flag.append(i)
                p.append(j)
            P.append(p)
        return Fs_p, L_p, Fs_f, L_f, Flag, P, np.array(Lab_p), np.array(Lab_f)

def getdata(file_name, seq_lens, train_ratio, batch_size):
    dataset = TrajDataset(file_name, seq_lens)
    len_data = len(dataset)
    train_nb_samples = int(len_data * train_ratio)
    test_nb_samples = len_data - train_nb_samples
    train_set, test_set = random_split(dataset, [train_nb_samples, test_nb_samples])
    return DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True), DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)