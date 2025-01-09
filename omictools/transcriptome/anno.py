import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import re


def SCINA(adata, signatures, max_iter=100, convergence_n=10, convergence_rate=0.99,
          sensitivity_cutoff=1, allow_unknown=False, log_file='SCINA.log'):
    """

    :param adata:表达数据，anndata格式
    :param signatures: 字典格式，键为字符串格式，值为字符串构成的列表，不包含空值与nan值例如：
        'Glial cell'：['S100a1', 'Mki67', 'Gfap', 'Vim', 'Tnc', 'Egfr', 'Plp1', 'Sox10']
    :param max_iter:最大迭代次数
    :param convergence_n:
    :param convergence_rate:
    :param sensitivity_cutoff:
    :param rm_overlap:
    :param allow_unknown:
    :param log_file:
    :return:
    """
    # 构建日志文件
    with open(log_file, 'w') as f:
        print('Start running SCINA.', file=f)
    # 构建状态日志文件
    status_file = f'{log_file}.status'
    # 统计所有标记基因
    all_sig = list(set([item for sublist in signatures.values() for item in sublist]))
    # 统计所有特征性低表达标记基因
    invert_sigs = [sig for sig in all_sig if re.match(r'^low_', sig)]
    # 提取表达量矩阵,并转换为稠密矩阵
    expr = adata.X.toarray()

    if invert_sigs:
        with open(log_file, 'a') as f:
            print('Converting expression matrix for low_genes.', file=f)
        # 去除特征低表达基因的字符串头"low_"
        invert_sigs = [sig for sig in invert_sigs if re.sub(r'^low_', '', sig) in adata.var_names]
        # 筛选高变的特征低表达基因
        invert_hv_sigs = list(set(invert_sigs) & set(adata.var_names))
        # 对低表达基因的表达量取负
        expr[:, adata.var_names.isin(invert_hv_sigs)] *= -1

    # 输入检查
    quality = check_inputs(expr, adata.var_names, signatures, max_iter, convergence_n, convergence_rate,
                           sensitivity_cutoff, log_file)

    if not quality['qual']:
        with open(log_file, 'a') as f:
            print('EXITING due to invalid parameters.', file=f)
        with open(status_file, 'w') as f:
            print('0', file=f)
        raise ValueError('SCINA stopped.')
    # 加载检查后的参数
    signatures = quality['sig']
    max_iter, convergence_n, convergence_rate, sensitivity_cutoff = quality['para']

    expr = expr.loc[[gene for sig in signatures for gene in sig]].copy()
    labels = np.zeros((adata.shape[1], convergence_n))
    unsatisfied = True

    if allow_unknown:
        tao = np.full(len(signatures), 1 / (len(signatures) + 1))
    else:
        tao = np.full(len(signatures), 1 / len(signatures))

    theta = []
    for i, sig in enumerate(signatures):
        expr = adata.loc[sig].values
        mean_high = np.percentile(expr, 70, axis=1)
        mean_low = np.percentile(expr, 30, axis=1)
        variances = np.var(expr, axis=1)
        sigma = np.diag(variances)

        theta.append({
            'mean': np.vstack([mean_high, mean_low]),
            'sigma1': sigma.copy(),
            'sigma2': sigma.copy()
        })

    def is_empty(matrix):
        return np.all(np.isnan(matrix) | (matrix == 0))

    theta = [t for t in theta if not is_empty(t['sigma1'])]

    sigma_min = min(min(sig['sigma1'].diagonal().min(), sig['sigma2'].diagonal().min()) for sig in theta) / 100

    while unsatisfied:
        prob_mat = np.tile(tao[:, np.newaxis], (1, adata.shape[1]))
        iter = 0
        labels_i = 0

        while iter < max_iter:
            iter += 1

            for i in range(len(signatures)):
                theta[i]['inverse_sigma1'] = inv(cholesky(theta[i]['sigma1']).T @ cholesky(theta[i]['sigma1']))
                theta[i]['inverse_sigma2'] = inv(cholesky(theta[i]['sigma2']).T @ cholesky(theta[i]['sigma2']))

            for r in range(prob_mat.shape[0]):
                expr = adata.loc[signatures[r]].values
                pdf_high = multivariate_normal.pdf(expr.T, theta[r]['mean'][0], theta[r]['sigma1'])
                pdf_low = multivariate_normal.pdf(expr.T, theta[r]['mean'][1], theta[r]['sigma2'])
                prob_mat[r] = tao[r] * pdf_high / (pdf_high + pdf_low)

            prob_mat = prob_mat / (1 - sum(tao) + prob_mat.sum(axis=0))

            tao = prob_mat.mean(axis=1)

            for i, sig in enumerate(signatures):
                expr = adata.loc[sig].values
                mean_high = (expr @ prob_mat[i]) / prob_mat[i].sum()
                mean_low = (expr @ (1 - prob_mat[i])) / (1 - prob_mat[i]).sum()

                keep = mean_high <= mean_low
                if keep.any():
                    mean_high[keep] = expr[keep].mean(axis=1)
                    mean_low[keep] = mean_high[keep]

                tmp1 = ((expr - mean_high[:, None]) ** 2).T
                tmp2 = ((expr - mean_low[:, None]) ** 2).T

                diag_sigma = (tmp1 @ prob_mat[i] + tmp2 @ (1 - prob_mat[i])) / prob_mat.shape[1]
                diag_sigma[diag_sigma < sigma_min] = sigma_min

                theta[i]['sigma1'] = np.diag(diag_sigma)
                theta[i]['sigma2'] = np.diag(diag_sigma)

            labels[:, labels_i] = np.argmax(np.vstack((1 - prob_mat.sum(axis=0), prob_mat)), axis=0) - 1

            if np.mean([len(set(labels[x])) == 1 for x in range(labels.shape[0])]) >= convergence_rate:
                with open(log_file, 'a') as f:
                    print('Job finished successfully.', file=f)
                with open(status_file, 'w') as f:
                    print('1', file=f)
                break

            labels_i = (labels_i + 1) % convergence_n
            if iter == max_iter:
                with open(log_file, 'a') as f:
                    print('Maximum iterations, breaking out.', file=f)

        dummytest = np.array([np.mean(theta[i]['mean'][0] - theta[i]['mean'][1] == 0) for i in range(len(signatures))])
        if np.all(dummytest <= sensitivity_cutoff):
            unsatisfied = False
        else:
            rev = np.where(dummytest > sensitivity_cutoff)[0]
            with open(log_file, 'a') as f:
                print(f'Remove dummy signatures: {rev}', file=f)
            signatures = [signatures[i] for i in range(len(signatures)) if i not in rev]
            tao = tao[np.isin(np.arange(len(tao)), rev, invert=True)]
            tao = tao / (1 - sum(tao) + sum(tao))
            theta = [theta[i] for i in range(len(theta)) if i not in rev]

    prob_mat = pd.DataFrame(prob_mat, index=['unknown'] + list(signatures), columns=adata.columns)
    labels = pd.DataFrame(labels, index=adata.columns)

    cell_labels = ['unknown'] + list(signatures)
    final_labels = [cell_labels[int(l)] for l in labels.iloc[:, -1]]

    return {
        'cell_labels': final_labels,
        'probabilities': prob_mat
    }


def check_inputs(exp, allgenes, signatures, max_iter, convergence_n, convergence_rate,
                 sensitivity_cutoff, log_file='SCINA.log'):
    """
    检查下列参数是否不含非空值
    :param exp:
    :param allgenes:
    :param signatures:
    :param max_iter:
    :param convergence_n:
    :param convergence_rate:
    :param sensitivity_cutoff:
    :param log_file:
    :return:
    """
    quality = 1

    # 检查表达矩阵中是否有空值
    if np.isnan(exp).sum():
        with open(log_file, 'a') as f:
            print('NA exists in expression matrix.', file=f)
        quality = 0

    # 检查signature中是否包含空值
    if any(not gene for sig in signatures.values() for gene in sig):
        with open(log_file, 'a') as f:
            print('Null cell type signature genes.', file=f)
        quality = 0
    else:
        # 去除基因集中的空值
        signatures = {k: [g for g in v if not g and g in allgenes] for k, v in signatures.items()}

        # 去除全0基因
        std_devs = exp.std(axis=1)
        signatures = {k: [g for g in v if std_devs[g] > 0] for k, v in signatures.items()}

    # 检查其他参数
    if pd.isna(convergence_n):
        with open(log_file, 'a') as f:
            print('Using convergence_n=10 by default', file=f)
        convergence_n = 10

    if pd.isna(max_iter):
        with open(log_file, 'a') as f:
            print('Using max_iter=1000 by default', file=f)
        max_iter = 1000
    elif max_iter < convergence_n:
        with open(log_file, 'a') as f:
            print('Using max_iter=convergence_n by default due to smaller than convergence_n.', file=f)
        max_iter = convergence_n

    if pd.isna(convergence_rate):
        with open(log_file, 'a') as f:
            print('Using convergence_rate=0.99 by default.', file=f)
        convergence_rate = 0.99

    if pd.isna(sensitivity_cutoff):
        with open(log_file, 'a') as f:
            print('Using sensitivity_cutoff=0.33 by default.', file=f)
        sensitivity_cutoff = 0.33

    return {
        'qual': quality,
        'sig': signatures,
        'para': [max_iter, convergence_n, convergence_rate, sensitivity_cutoff]
    }