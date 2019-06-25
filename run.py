import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture


def main():
    # ============================================
    # === Loading data
    # ============================================
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')

    cols = [c for c in train.columns if c not in ['id', 'target']]
    cols.remove('wheezy-copper-turtle-magic')

    # ============================================
    # === Step 1 - Build first QDA model and predict test
    # ============================================
    # initialize variables
    oof = np.zeros(len(train))
    preds = np.zeros(len(test))

    # build 512 separate models
    for i in range(512):
        # only train with data where wheezy equals i
        train2 = train[train['wheezy-copper-turtle-magic'] == i]
        test2 = test[test['wheezy-copper-turtle-magic'] == i]
        idx1 = train2.index
        idx2 = test2.index
        train2.reset_index(drop=True, inplace=True)

        # feature selection (use approx 40 of 255 features)
        sel = VarianceThreshold(threshold=1.5).fit(train2[cols])
        train3 = sel.transform(train2[cols])
        test3 = sel.transform(test2[cols])

        # stratified k-fold
        skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)
        for train_index, test_index in skf.split(train3, train2['target']):
            # model and predict with QDA
            clf = QuadraticDiscriminantAnalysis(reg_param=0.5)
            clf.fit(train3[train_index, :], train2.loc[train_index]['target'])
            oof[idx1[test_index]] = clf.predict_proba(train3[test_index, :])[:, 1]
            preds[idx2] += clf.predict_proba(test3)[:, 1] / skf.n_splits

    # print cv auc
    auc = roc_auc_score(train['target'], oof)
    print('QDA scores CV =', round(auc, 5))

    # ============================================
    # === Step2 - Add pseudo label data and build second model
    # ============================================
    n_init = 10
    test['target'] = preds

    # initialize variables
    oof = np.zeros(len(train))
    preds = np.zeros(len(test))

    # build 512 separate models
    for k in range(512):
        # only train with data where wheezy equals i
        train2 = train[train['wheezy-copper-turtle-magic'] == k]
        train2p = train2.copy()
        idx1 = train2.index
        test2 = test[test['wheezy-copper-turtle-magic'] == k]

        # add pseudo labeled data
        test2p = test2[(test2['target'] <= 0.01) | (test2['target'] >= 0.99)].copy()
        test2p.loc[test2p['target'] >= 0.5, 'target'] = 1
        test2p.loc[test2p['target'] < 0.5, 'target'] = 0
        train2p = pd.concat([train2p, test2p], axis=0)
        train2p.reset_index(drop=True, inplace=True)

        # feature selextion (use approx 40 of 255 features)
        sel = VarianceThreshold(threshold=1.5).fit(train2p[cols])
        train3p = sel.transform(train2p[cols])
        train3 = sel.transform(train2[cols])
        test3 = sel.transform(test2[cols])

        # get cluster labels
        target_0 = np.argwhere(train2p["target"].values == 0).reshape(-1)
        target_1 = np.argwhere(train2p["target"].values == 1).reshape(-1)
        n_cols = train3.shape[1]
        # 1引くのは万が一insertがうまく言っていなかったとき対策
        cluster_labels = np.zeros_like(train2p["target"].values) - 1
        proba_x_0 = np.zeros((len(target_0), n_cols * 2))
        proba_x_1 = np.zeros((len(target_1), n_cols * 2))

        # calculate GMM per col
        for j in range(n_cols):
            # target = 0
            kms_0 = GaussianMixture(
                n_components=2, max_iter=10000, n_init=n_init,
                means_init=[[-1], [1]], init_params="kmeans"
            )
            kms_0.fit(train3p[target_0, j:j + 1])
            pred_0 = kms_0.predict_proba(train3p[target_0, j:j + 1])
            proba_x_0[:, j * 2:(j + 1) * 2] = pred_0

            # target = 1
            kms_1 = GaussianMixture(
                n_components=2, max_iter=10000, n_init=n_init,
                means_init=[[-1], [1]], init_params="kmeans"
            )
            kms_1.fit(train3p[target_1, j:j + 1])
            pred_1 = kms_1.predict_proba(train3p[target_1, j:j + 1])
            proba_x_1[:, j * 2:(j + 1) * 2] = pred_1

        # re-calculate GMM
        kms_0 = GaussianMixture(
            n_components=3, max_iter=10000, n_init=n_init, init_params="kmeans",
        )
        kms_0.fit(proba_x_0)

        kms_1 = GaussianMixture(
            n_components=3, max_iter=10000, n_init=n_init, init_params="kmeans",
        )
        kms_1.fit(proba_x_1)

        # predict cluster labels
        cluster_labels_0 = kms_0.predict(proba_x_0)
        cluster_labels_1 = kms_1.predict(proba_x_1) + 3
        cluster_labels[target_0] = cluster_labels_0
        cluster_labels[target_1] = cluster_labels_1

        # stratified k-fold
        skf = StratifiedKFold(n_splits=11, random_state=42, shuffle=True)
        for train_index, test_index in skf.split(train3p, cluster_labels):
            test_index3 = test_index[test_index < len(train3)]   # ignore pseudo in oof

            # model and predict with QDA
            clf = QuadraticDiscriminantAnalysis(reg_param=0.5)
            clf.fit(train3p[train_index, :], cluster_labels[train_index])

            val_prediction_6 = clf.predict_proba(train3p[test_index3, :])
            val_prediction = val_prediction_6[:, 3] + val_prediction_6[:, 4] + val_prediction_6[:, 5]
            oof[idx1[test_index3]] = val_prediction
            test_prediction_6 = clf.predict_proba(test3)
            test_prediction = test_prediction_6[:, 3] + test_prediction_6[:, 4] + test_prediction_6[:, 5]
            preds[test2.index] += test_prediction / skf.n_splits

    # print cv auc
    auc = roc_auc_score(train['target'], oof)
    print('Pseudo Labeled QDA scores CV =', round(auc, 5))

    # ============================================
    # === Make Submission
    # ============================================
    sub = pd.read_csv('../input/sample_submission.csv')
    sub['target'] = preds
    sub.to_csv('submission.csv',index=False)


if __name__ == '__main__':
    main()
