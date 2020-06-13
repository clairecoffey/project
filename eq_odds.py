
"""
This equalised odds implementation is taken from https://github.com/gpleiss/equalized_odds_and_calibration.
From the paper "On Fairness and Calibration" (http://papers.nips.cc/paper/7151-on-fairness-and-calibration.pdf).
Altered slightly to fit our code, but all credit to the original authors. 
"""
import cvxpy as cvx
import numpy as np
from collections import namedtuple

class Model(namedtuple('Model', 'pred label')):

    def warn(*args, **kwargs):
        pass

    def logits(self):
        raw_logits = np.clip(np.log(self.pred / (1 - self.pred)), -100, 100)
        return raw_logits

    def num_samples(self):
        return len(self.pred)

    def base_rate(self):
        """
        Percentage of samples belonging to the positive class
        """
        return np.mean(self.label)

    def accuracy(self):
        return self.accuracies().mean()

    def precision(self):
        return (self.label[self.pred.round() == 1]).mean()

    def recall(self):
        return (self.label[self.label == 1].round()).mean()

    def tpr(self):
        """
        True positive rate
        """
        return np.mean(np.logical_and(self.pred.round() == 1, self.label == 1))

    def fpr(self):
        """
        False positive rate
        """
        return np.mean(np.logical_and(self.pred.round() == 1, self.label == 0))

    def tnr(self):
        """
        True negative rate
        """
        return np.mean(np.logical_and(self.pred.round() == 0, self.label == 0))

    def fnr(self):
        """
        False negative rate
        """
        return np.mean(np.logical_and(self.pred.round() == 0, self.label == 1))

    def fn_cost(self):
        """
        Generalized false negative cost
        """
        return 1 - self.pred[self.label == 1].mean()

    def fp_cost(self):
        """
        Generalized false positive cost
        """
        return self.pred[self.label == 0].mean()

    def accuracies(self):
        return self.pred.round() == self.label

    def eq_odds(self, othr, mix_rates=None):
        has_mix_rates = not (mix_rates is None)
        if not has_mix_rates:
            mix_rates = self.eq_odds_optimal_mix_rates(othr)
        sp2p, sn2p, op2p, on2p = tuple(mix_rates)

        self_fair_pred = self.pred.copy()
        self_pp_indices, = np.nonzero(self.pred.round())
        self_pn_indices, = np.nonzero(1 - self.pred.round())
        np.random.shuffle(self_pp_indices)
        np.random.shuffle(self_pn_indices)

        n2p_indices = self_pn_indices[:int(len(self_pn_indices) * sn2p)]
        self_fair_pred[n2p_indices] = 1 - self_fair_pred[n2p_indices]
        p2n_indices = self_pp_indices[:int(len(self_pp_indices) * (1 - sp2p))]
        self_fair_pred[p2n_indices] = 1 - self_fair_pred[p2n_indices]

        othr_fair_pred = othr.pred.copy()
        othr_pp_indices, = np.nonzero(othr.pred.round())
        othr_pn_indices, = np.nonzero(1 - othr.pred.round())
        np.random.shuffle(othr_pp_indices)
        np.random.shuffle(othr_pn_indices)

        n2p_indices = othr_pn_indices[:int(len(othr_pn_indices) * on2p)]
        othr_fair_pred[n2p_indices] = 1 - othr_fair_pred[n2p_indices]
        p2n_indices = othr_pp_indices[:int(len(othr_pp_indices) * (1 - op2p))]
        othr_fair_pred[p2n_indices] = 1 - othr_fair_pred[p2n_indices]

        fair_self = Model(self_fair_pred, self.label)
        fair_othr = Model(othr_fair_pred, othr.label)

        if not has_mix_rates:
            return fair_self, fair_othr, mix_rates
        else:
            return fair_self, fair_othr

    def eq_odds_optimal_mix_rates(self, othr):
        sbr = float(self.base_rate())
        obr = float(othr.base_rate())

        sp2p = cvx.Variable(1)
        sp2n = cvx.Variable(1)
        sn2p = cvx.Variable(1)
        sn2n = cvx.Variable(1)

        op2p = cvx.Variable(1)
        op2n = cvx.Variable(1)
        on2p = cvx.Variable(1)
        on2n = cvx.Variable(1)

        sfpr = self.fpr() * sp2p + self.tnr() * sn2p
        sfnr = self.fnr() * sn2n + self.tpr() * sp2n
        ofpr = othr.fpr() * op2p + othr.tnr() * on2p
        ofnr = othr.fnr() * on2n + othr.tpr() * op2n
        error = sfpr + sfnr + ofpr + ofnr

        sflip = 1 - self.pred
        sconst = self.pred
        oflip = 1 - othr.pred
        oconst = othr.pred

        sm_tn = np.logical_and(self.pred.round() == 0, self.label == 0)
        sm_fn = np.logical_and(self.pred.round() == 0, self.label == 1)
        sm_tp = np.logical_and(self.pred.round() == 1, self.label == 1)
        sm_fp = np.logical_and(self.pred.round() == 1, self.label == 0)

        om_tn = np.logical_and(othr.pred.round() == 0, othr.label == 0)
        om_fn = np.logical_and(othr.pred.round() == 0, othr.label == 1)
        om_tp = np.logical_and(othr.pred.round() == 1, othr.label == 1)
        om_fp = np.logical_and(othr.pred.round() == 1, othr.label == 0)

        spn_given_p = (sn2p * (sflip * sm_fn).mean() + sn2n * (sconst * sm_fn).mean()) / sbr + \
                      (sp2p * (sconst * sm_tp).mean() + sp2n * (sflip * sm_tp).mean()) / sbr

        spp_given_n = (sp2n * (sflip * sm_fp).mean() + sp2p * (sconst * sm_fp).mean()) / (1 - sbr) + \
                      (sn2p * (sflip * sm_tn).mean() + sn2n * (sconst * sm_tn).mean()) / (1 - sbr)

        opn_given_p = (on2p * (oflip * om_fn).mean() + on2n * (oconst * om_fn).mean()) / obr + \
                      (op2p * (oconst * om_tp).mean() + op2n * (oflip * om_tp).mean()) / obr

        opp_given_n = (op2n * (oflip * om_fp).mean() + op2p * (oconst * om_fp).mean()) / (1 - obr) + \
                      (on2p * (oflip * om_tn).mean() + on2n * (oconst * om_tn).mean()) / (1 - obr)

        constraints = [
            sp2p == 1 - sp2n,
            sn2p == 1 - sn2n,
            op2p == 1 - op2n,
            on2p == 1 - on2n,
            sp2p <= 1,
            sp2p >= 0,
            sn2p <= 1,
            sn2p >= 0,
            op2p <= 1,
            op2p >= 0,
            on2p <= 1,
            on2p >= 0,
            spp_given_n == opp_given_n,
            spn_given_p == opn_given_p,
        ]

        prob = cvx.Problem(cvx.Minimize(error), constraints)
        prob.solve()

        res = np.array([sp2p.value, sn2p.value, op2p.value, on2p.value])
        return res

    def __repr__(self):
        return '\n'.join([
            'Accuracy:\t%.3f' % self.accuracy(),
            'F.P. cost:\t%.3f' % self.fp_cost(),
            'F.N. cost:\t%.3f' % self.fn_cost(),
            'Base rate:\t%.3f' % self.base_rate(),
            'Avg. score:\t%.3f' % self.pred.mean(),
        ])


if __name__ == '__main__':

    import pandas as pd
    import sys

    if not len(sys.argv) == 2:
        raise RuntimeError('Invalid number of arguments')

    # Load the validation set scores from csvs
    data_filename = sys.argv[1]
    test_and_val_data = pd.read_csv(sys.argv[1])

    #  split the data into two equal size sets - one for computing the fairness constants
    # order = np.random.permutation(len(test_and_val_data))
    #put test data first so indexes match up with original ones
    # used for cross-referencing data to find misclassified individuals later 
    test_data = test_and_val_data.iloc[:(int(len(test_and_val_data)/2)),:]
    val_data = test_and_val_data.iloc[(int(len(test_and_val_data)/2)):,:]

    # Create model objects - one for each group, validation and test
    group_0_val_data = val_data[val_data['group'] == 0]
    group_1_val_data = val_data[val_data['group'] == 1]
    group_0_test_data = test_data[test_data['group'] == 0]
    group_1_test_data = test_data[test_data['group'] == 1]

    group_0_val_model = Model(group_0_val_data['prediction'].values, group_0_val_data['label'].values)
    group_1_val_model = Model(group_1_val_data['prediction'].values, group_1_val_data['label'].values)
    group_0_test_model = Model(group_0_test_data['prediction'].values, group_0_test_data['label'].values)
    group_1_test_model = Model(group_1_test_data['prediction'].values, group_1_test_data['label'].values)

    # Find mixing rates for equalized odds models
    _, _, mix_rates = Model.eq_odds(group_0_val_model, group_1_val_model)

    # Apply the mixing rates to the test models
    eq_odds_group_0_test_model, eq_odds_group_1_test_model= Model.eq_odds(group_0_test_model,
                                                                           group_1_test_model,
                                                                           mix_rates)

    # Print results on test model    
    # print('Original group 0 model:\n%s\n' % repr(group_0_test_model))
    # print('Predictions: ')
    # print(group_0_test_model.pred)
    # print('True labels: ') 
    # print(group_0_test_model.label)
    # print('Original group 1 model:\n%s\n' % repr(group_1_test_model))
    # print('Predictions: ')
    # print(group_1_test_model.pred)
    # print('True labels: ') 
    # print(group_1_test_model.label)
    # print('Equalized odds group 0 model:\n%s\n' % repr(eq_odds_group_0_test_model))
    # print('Predictions: ')
    # print(eq_odds_group_0_test_model.pred)
    # print('True labels: ') 
    # print(eq_odds_group_0_test_model.label)
    # print('Equalized odds group 1 model:\n%s\n' % repr(eq_odds_group_1_test_model))
    # print('Predictions: ')
    # print(eq_odds_group_1_test_model.pred)
    # print('True labels: ') 
    # print(eq_odds_group_1_test_model.label)

    #export equalised odds predictions to csv
    # group_0 = (pd.DataFrame([eq_odds_group_0_test_model.pred, eq_odds_group_0_test_model.label]))
    # group_1 = (pd.DataFrame([eq_odds_group_1_test_model.pred, eq_odds_group_1_test_model.label]))
    group_0 = pd.DataFrame({ 'predictions':eq_odds_group_0_test_model.pred })
    group_1 = pd.DataFrame({ 'predictions':eq_odds_group_1_test_model.pred })
    group_0 = group_0.assign(true_labels=pd.Series(eq_odds_group_0_test_model.label))
    group_1 = group_1.assign(true_labels=pd.Series(eq_odds_group_1_test_model.label))

    # group_0 = pd.pivot_table(group_0, index='index' columns = group_0.loc[:], values=)
    group_0.to_csv(r'group_0.csv', index=False)
    group_1.to_csv(r'group_1.csv', index=False)
    # np.savetxt("eq_odds_pred_group_0.csv", eq_odds_group_0_test_model.pred, delimiter=",")
    # np.savetxt("eq_odds_pred_group_1.csv", eq_odds_group_1_test_model.pred, delimiter=",")
