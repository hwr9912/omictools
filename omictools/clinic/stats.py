# -*- coding: UTF-8 -*-
# Authorized by Vlon Jang
# Created on 2022/6/14
# Home: www.medai.icu
# Email: wangqingbaidu@gmail.com
# Copyright 2015-2022 All Rights Reserved.

import copy
from typing import List, Union

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import chi2, kstest
from scipy.stats import ttest_ind, mannwhitneyu, chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# from statsmodels.formula.api import ols
# from statsmodels.stats.anova import anova_lm
from scipy.stats import shapiro


def clinic_stats(data: pd.DataFrame,
                 stats_columns: list[str],
                 label_column: str = "label",
                 group_column: str = "group",
                 cohort_column: str = "cohort",
                 continuous_columns: list[str] = None,
                 pretty: bool = True,
                 save_table: bool = True,
                 verbose: bool = False,
                 **kwargs) -> Union[dict, DataFrame, None]:
    """

    :param data: 待统计数据，至少包含标签列，分组列
    :param stats_columns: 待统计的列
    :param label_column: 标签列名
    :param group_column: 分组列名，一般是train或者val
    :param cohort_column: 研究队列名
    :param continuous_columns: 连续值的列
    :param pretty: 是否格式化输出
    :param verbose: 是否打印log
    :param kwargs:
    :return:
    """
    # 清空输入数据的index
    data = data.reset_index(drop=True)
    b_test = None
    p_test = None
    data_columns = list(data.columns)
    # 判断label列是否在数据中
    if label_column not in data_columns:
        if any(c.startswith(label_column) for c in data_columns):
            raise ValueError(f"你的数据里面存在多列为{label_column}，请确保特征文件没有多余的{label_column}列.")
        else:
            raise ValueError(f"你的数据里面没有{label_column}列，请仔细检查数据，大概率是你的label文件里面有缺失值。")

    # 二分类标签+存在连续变量
    if len(data[label_column].unique()) == 2 and continuous_columns is not None:
        sel_columns = [c for c in set(continuous_columns + [label_column, group_column]) if c is not None]
        b_test = clinic_stats_dual(data[sel_columns],
                                   stats_columns=continuous_columns,
                                   label_column=label_column,
                                   group_column=group_column,
                                   continuous_columns=continuous_columns,
                                   pretty=pretty,
                                   verbose=verbose,
                                   **kwargs)
        stats_columns = [c for c in stats_columns if c not in continuous_columns]
        continuous_columns = None

    # 多分类标签
    if stats_columns:
        p_test = clinic_stats_pluralism(data, stats_columns=stats_columns, label_column=label_column,
                                        group_column=group_column, continuous_columns=continuous_columns,
                                        pretty=pretty, verbose=verbose, **kwargs)

    # (暂时意义不明）
    if b_test is not None and p_test is not None:
        if pretty:
            # print(b_test, p_test)
            return pd.concat([b_test, p_test], axis=0)
        else:
            return {**b_test, **p_test}
    if b_test is not None:
        return b_test
    else:
        return p_test


def clinic_stats_dual(sdata: DataFrame,
                      stats_columns: Union[str, List[str]],
                      label_column: str ='label',
                      group_column: Union[str, List[str]] = 'group',
                      continuous_columns: Union[str, List[str]] = None,
                      pretty: bool = True,
                      save_table: bool = True,
                      label_spec: bool = True,
                      test_type: str = None,
                      verbose: bool = False) -> Union[dict, DataFrame]:
    """

    :param sdata: 数据
    :param stats_columns: 需要统计的列名
    :param label_column: 二分类的标签列，默认`label`
    :param group_column: 分组统计依据，例如区分训练组、测试组、验证组。
    :param cohort_column: 研究中心队列名称
    :param continuous_columns: 那些列是连续变量，连续变量统计均值方差。
    :param pretty: 是否对结果进行格式美化。
    :param label_spec: 是否使用每个细分label的统计。默认计算。
    :param test_type: 测试类型，目前支持Ttest和Utest。
    :param verbose: 是否打印选择的统计手段。
    :return:
    """
    data = copy.deepcopy(sdata)
    # stats_columns = _c2list(stats_columns)
    # continuous_columns = _c2list(continuous_columns)

    for fn in stats_columns + continuous_columns + [label_column, group_column]:
        if fn and fn not in data.columns:
            raise ValueError(f"{fn}没有在{list(data.columns)}中。请检查设置！")
    if label_column is None:
        raise ValueError('标签列label_column不能为None!')
    ulabels = data[label_column].unique()
    if len(ulabels) != 2:
        raise ValueError(
            f'此接口只能用于2元结果类型的显著性检测，现在{len(ulabels)}元，他们是:{ulabels}，如果是多元显著性检测可以'
            f'使用`clinic_stats_chi_square`接口')
    if test_type is not None:
        # logger.warning(f'test_type={test_type}参数自行设置，Onekey将不进行正态分布检验。')
        print(f'test_type={test_type}参数自行设置，Onekey将不进行正态分布检验。')
    # test_type = test_type.lower()
    # if test_type not in ['ttest', 'utest']:
    #     raise ValueError(f"不支持的检测类型{test_type}，只支持ttest，utest")
    if group_column is not None:
        if group_column in stats_columns:
            raise ValueError(f'分组列{group_column}不能在分析列{stats_columns}中')
        data[group_column] = data[group_column].astype(str)
        groups = sorted(data[group_column].unique())
        if 'train' in groups and 'test' in groups and 'val' in groups:
            groups = ['train', 'val', 'test']
        elif 'train' in groups and 'test' in groups:
            groups = ['train', 'test'] + list(set(groups) - {'train', 'test'})
        # print(groups)
        data = {g: data[data[group_column] == g] for g in groups}
    else:
        data = {'': data}
        groups = ['']

    stats = {}
    for col in stats_columns:
        stats[col] = {}
        # 此处data是字典格式，键是group变量里的分组，值是对应组下的dataframe
        for group, df in data.items():
            # 1. Shapiro-Wilk 正态性检验
            shapiro_results = shapiro(df[col])

            # 2. 根据 Shapiro 检验选择检验方法
            test_type = 'ttest' if shapiro_results.pvalue > 0.05 else 'utest'

            # 3. 检查标签类别数量并进行统计检验
            labels = sdata[label_column].unique()
            # 对于二分类变量进行检验
            if len(labels) == 2:
                # 通过正态性检验的使用t检验
                if test_type == 'ttest':
                    ttest_results = ttest_ind(df[df[label_column] == labels[0]][col],
                                              df[df[label_column] == labels[1]][col])
                    p_value = ttest_results.pvalue
                # 未通过正态性检验的使用Mann-Whitney U 检验（也称为 Wilcoxon 秩和检验或 Mann-Whitney-Wilcoxon 检验）
                else:
                    utest_results = mannwhitneyu(df[df[label_column] == labels[0]][col],
                                                 df[df[label_column] == labels[1]][col])
                    p_value = utest_results.pvalue

                p_value = '<0.001' if p_value < 1e-3 else p_value
            else:
                # logger.warning(f'特征:{col}，在group:{g}，存在{len(labels)}(≠2)个不同类别，无法进行T检验')
                p_value = None

            # 4. 处理离散变量
            if col not in continuous_columns:
                s = {k: {'cnt': v, 'ratio': v / df.shape[0]} for k, v in
                     sorted(df[col].value_counts().items(), key=lambda x: x[0])}
                s.update({ak: {'cnt': 0, 'ratio': 0} for ak in sdata[col].value_counts().keys() if ak not in s})
                for label in labels:
                    l_data = df[df[label_column] == label]
                    for k, v in sorted(l_data[col].value_counts().items(), key=lambda x: x[0]):
                        s[k].update({f"label={label}": {'cnt': v, 'ratio': v / l_data.shape[0]}})
                s['__pvalue__'] = p_value
            else:
                s = {'mean': df[col].mean(), 'std': df[col].std(), '__pvalue__': p_value}
                for label in labels:
                    l_data = df[df[label_column] == label][col]
                    s.update({f"mean | label={label}": l_data.mean(), f'std | label={label}': l_data.std()})

            # 5. 存储统计结果
            stats[col][group] = s
    # return json.dumps(stats, indent=True)
    return pretty_stats(stats, ulabels if label_spec else [], groups, continuous_columns, precision=2) if pretty else stats
    # return stats

def pretty_stats(stats, labels, groups, continuous_columns, precision):
    title = ['feature_name']
    labels = sorted(labels)
    for group in groups:
        title.extend(f"{group}-label={label}" for label in ['ALL'] + labels)
        title.append('pvalue')
    c_stats = []
    d_stats = []
    for k, v in stats.items():
        group_lines = [k]
        if k in continuous_columns:
            for g, stat in v.items():
                mean_keys = ['mean'] + [f"mean | label={label}" for label in labels]
                std_keys = ['std'] + [f"std | label={label}" for label in labels]
                for mk, sk in zip(mean_keys, std_keys):
                    if mk in stat and sk in stat:
                        fmt = f"{{mk:.{precision}f}}±{{sk:.{precision}f}}"
                        # group_lines.append(f"{stat[mk]:.4f}±{stat[sk]:.4f}")
                        group_lines.append(fmt.format(mk=stat[mk], sk=stat[sk]))
                    else:
                        group_lines.append('null')
                group_lines.append(stat['__pvalue__'])
            c_stats.append(group_lines)
        else:
            for g, stat in v.items():
                group_lines.extend([''] * (len(labels) + 1))
                group_lines.append(stat['__pvalue__'])
            d_stats.append(group_lines)

            cnt_keys = [f"label={label}" for label in labels]
            addition_lines = {}
            for g, stat in v.items():
                for s_k, s_v in stat.items():
                    if s_k != '__pvalue__':
                        fmt = f"{{cnt}}({{ratio:.{precision}f}})"
                        if s_k in addition_lines:
                            addition_lines[s_k].extend([fmt.format(cnt=s_v['cnt'], ratio=s_v['ratio'] * 100)])
                        else:
                            addition_lines[s_k] = [s_k, fmt.format(cnt=s_v['cnt'], ratio=s_v['ratio'] * 100)]
                        # Add addition lines
                        for ck in cnt_keys:
                            if ck in s_v:
                                addition_lines[s_k].append(fmt.format(cnt=s_v[ck]['cnt'], ratio=s_v[ck]['ratio'] * 100))

                            else:
                                addition_lines[s_k].append('null')
                        addition_lines[s_k].append('')
            d_stats.extend([v for v in addition_lines.values()])
    # print(c_stats + d_stats)
    return pd.DataFrame(c_stats + d_stats, columns=title)

# def _c2list(a):
#     if a is None:
#         return []
#     if not isinstance(a, (list, tuple)):
#         return [a]
#     else:
#         return a
#
# def _map_p_value(p):
#     if p == 0:
#         p_value = ''
#     elif p < 1e-3:
#         p_value = '<0.001'
#         # p_value = 1e-6
#     else:
#         p_value = p
#     return p_value
#
#
# def concat_mean_std(mean, std, decimals=None, not_concat=None):
#     decimals = decimals or get_param_in_cwd('display.precision', 3)
#     fmt = f"{{value:.{decimals}f}}"
#     if isinstance(not_concat, (list, tuple)):
#         not_concat = [not_concat]
#     data_columns = [c for c in mean.columns if c not in not_concat]
#     if isinstance(mean, pd.DataFrame):
#         mean_data = mean[data_columns]
#         std_data = std[data_columns]
#     else:
#         mean_data = mean
#         std_data = std
#
#     assert mean_data.shape == std_data.shape
#     x, y = mean_data.shape
#     data = []
#     for i in range(x):
#         data_ = []
#         for j in range(y):
#             data_.append(f"{fmt.format(value=mean_data.iloc[i][j])} ± {fmt.format(value=std_data.iloc[i][j])}")
#         data.append(data_)
#     data = pd.DataFrame(data, index=mean.index, columns=data_columns)
#     data = pd.concat([data, mean[not_concat]], axis=1)
#     return data
#
#
#
# def clinic_stats_pluralism(sdata: DataFrame, stats_columns: Union[str, List[str]], label_column='label',
#                            group_column: str = None, continuous_columns: Union[str, List[str]] = None,
#                            pretty: bool = True, label_spec: bool = True, auto_correct: bool = False,
#                            verbose: bool = False, **kwargs) -> Union[dict, DataFrame]:
#     """
#     多元类别的统计分析，所有stats_columns中为离散数值的都会被使用卡方检验，所有连续数值的都被会使用ANOVA（方差分析）进行显著性检验。
#
#     Args:
#         sdata: 数据
#         stats_columns: 需要统计的列名
#         label_column: 二分类的标签列，默认`label`
#         group_column: 分组统计依据，例如区分训练组、测试组、验证组。
#         continuous_columns: 那些列是连续变量，连续变量统计均值方差。
#         pretty: bool, 是否对结果进行格式美化。
#         label_spec: bool, 是否使用每个细分label的统计。默认计算。
#         auto_correct: bool, 是否进行自动校准， 默认为否
#         verbose: bool, 是否打印日志
#
#     Returns:
#         stats DataFrame or json
#
#     """
#     data = copy.deepcopy(sdata)
#     stats_columns = _c2list(stats_columns)
#     continuous_columns = _c2list(continuous_columns)
#     for fn in stats_columns + [label_column, group_column]:
#         if fn and fn not in data.columns:
#             raise ValueError(f"{fn}没有在{list(data.columns)}中。请检查设置！")
#     if label_column is None:
#         raise ValueError('标签列label_column不能为None!')
#     ulabels = data[label_column].unique()
#     for c in stats_columns:
#         if not np.int8 <= data[c].dtype <= np.int64 and c not in continuous_columns and auto_correct:
#             logger.warning(f'特征{c}设置错误，被强制设置为连续特征。')
#             continuous_columns.append(c)
#     if group_column is not None:
#         if group_column in stats_columns:
#             raise ValueError(f'分组列{group_column}不能在分析列{stats_columns}中')
#         data[group_column] = data[group_column].astype(str)
#         groups = sorted(data[group_column].unique())
#         if 'train' in groups and 'test' in groups and 'val' in groups:
#             groups = ['train', 'val', 'test']
#         elif 'train' in groups and 'test' in groups:
#             groups = ['train', 'test']
#         # print(groups)
#         data = {g: data[data[group_column] == g] for g in groups}
#     else:
#         data = {'': data}
#         groups = ['']
#
#     stats = {}
#     for c in stats_columns:
#         stats[c] = {}
#         for g, d in data.items():
#             # Compute p_value
#             labels = sdata[label_column].unique()
#             if c not in continuous_columns:
#                 value_counts = []
#                 sub_labels = list(d[c].unique())
#                 for l in labels:
#                     value_count = [0] * len(sub_labels)
#                     for k_, v_ in d[d[label_column] == l][c].value_counts().items():
#                         value_count[sub_labels.index(k_)] = v_
#                     value_counts.append(value_count)
#                 chi2_value, p_value, *_ = chi2_contingency(value_counts)
#                 s = {k: {'cnt': v, 'ratio': v / d.shape[0]}
#                      for k, v in sorted(d[c].value_counts().items(), key=lambda x: x[0])}
#                 s.update({ak: {'cnt': 0, 'ratio': 0} for ak in sdata[c].value_counts().keys() if ak not in s})
#                 for label in labels:
#                     l_data = d[d[label_column] == label]
#                     for k, v in sorted(l_data[c].value_counts().items(), key=lambda x: x[0]):
#                         s[k].update({f"label={label}": {'cnt': v, 'ratio': v / l_data.shape[0]}})
#                 s['__pvalue__'] = _map_p_value(p_value)
#                 if verbose:
#                     print(f'Feature {c} using chi2, chi2: {chi2_value}, p_value: {p_value}...')
#             else:
#                 if verbose:
#                     print(f'Feature {c} Using anova...')
#                 model = ols(rf'{c} ~ {label_column}', d[[c, label_column]])
#                 p_value = anova_lm(model.fit()).loc[label_column]['PR(>F)']
#                 s = {'mean': d[c].mean(), 'std': d[c].std(), '__pvalue__': _map_p_value(p_value)}
#                 for label in labels:
#                     l_data = d[d[label_column] == label][c]
#                     s.update({f"mean | label={label}": l_data.mean(), f'std | label={label}': l_data.std()})
#             stats[c][g] = s
#
#     return pretty_stats(stats, ulabels if label_spec else [], groups, continuous_columns) if pretty else stats
#
#
# def hosmer_lemeshow_test(y_true, y_pred, bins=10, remap: bool = False, **kwargs):
#     """
#     Args:
#         y_true: ground_truth label name is y
#         y_pred: prediction value column name is y_hat
#         bins: Number of bins
#
#     Returns:
#     """
#     if remap:
#         if kwargs.get('EX', None) is None:
#             model = LogisticRegression()
#         else:
#             model = RandomForestClassifier(**kwargs.get('EX'))
#         if len(y_pred.shape) == 1:
#             y_pred = np.reshape(np.array(y_pred), [-1, 1])
#             model.fit(y_pred, y_true)
#         else:
#             model.fit(y_pred, y_true)
#         y_pred = model.predict_proba(y_pred)[:, 1]
#         # y_pred = np.array(normalize_df(pd.DataFrame(y_pred), method='minmax'))
#     data = pd.DataFrame(zip(y_true.ravel(), y_pred.ravel()), columns=['y', 'y_hat'])
#     data = data.sort_values('y_hat')
#     data['Q_group'] = pd.qcut(data['y_hat'], bins, duplicates='drop')
#
#     y_p = data['y'].groupby(data.Q_group).sum()
#     y_total = data['y'].groupby(data.Q_group).count()
#     y_n = y_total - y_p
#
#     y_hat_p = data['y_hat'].groupby(data.Q_group).sum()
#     y_hat_total = data['y_hat'].groupby(data.Q_group).count()
#     y_hat_n = y_hat_total - y_hat_p
#
#     hltest = (((y_p - y_hat_p) ** 2 / y_hat_p) + ((y_n - y_hat_n) ** 2 / y_hat_n)).sum()
#     pval = 1 - chi2.cdf(hltest, bins - 2)
#
#     return pval
#
#
# def map2numerical(data: pd.DataFrame, mapping_columns: Union[str, List[str]], inplace=True):
#     mapping = {}
#     if not inplace:
#         new_data = copy.deepcopy(data)
#     else:
#         new_data = data
#     if not isinstance(mapping_columns, list):
#         mapping_columns = [mapping_columns]
#     assert all(c in data.columns for c in mapping_columns)
#     for c in mapping_columns:
#         unique_labels = {v: idx for idx, v in enumerate(sorted(np.unique(np.array(data[c]))))}
#         mapping[c] = unique_labels
#         new_data[[c]] = new_data[[c]].applymap(lambda x: unique_labels[x])
#     return new_data, mapping


if __name__ == '__main__':
    # from onekey_algo import OnekeyDS as okds
    #
    # test_data = pd.read_csv(okds.survival)
    # val_ = int(test_data.shape[0] * 0.2)
    # test_data['group'] = ['train'] * (test_data.shape[0] - val_) + ['val'] * val_
    # print(json.dumps(clinic_stats_dual(test_data, stats_columns=['gender', 'age', 'Tstage', 'smoke', 'BMI'],
    #                                    pretty=False,
    #                                    label_column='result', group_column='group',
    #                                    continuous_columns=['age', 'BMI']),
    #                  ensure_ascii=False, indent=True))
    # print(clinic_stats_dual(test_data, stats_columns=['gender', 'age', 'Tstage', 'smoke', 'BMI'],
    #                         label_column='result', group_column=None,
    #                         continuous_columns=['age', 'BMI'], pretty=True,
    #                         test_type='utest'))
    # print(clinic_stats(test_data, stats_columns=['gender', 'age', 'Tstage', 'smoke', 'BMI'],
    #                    label_column='result', group_column='group',
    #                    continuous_columns=['age', 'BMI'], pretty=True))
    #
    # print(json.dumps(,ensure_ascii = False, indent = True))

    # a = np.random.rand(100)
    # print(hosmer_lemeshow_test(a, a))
    import os
    import json

    os.chdir(r"D:\Python\machine_learning\SAH_PHH\wh_comp")
    example = pd.read_csv(r"data\clinical.csv")
    example = example.drop(["group"], axis=1)

    stats = clinic_stats_dual(sdata=example,
                              stats_columns=list(example.columns[4:]),
                              label_column='label',
                              group_column='cohort',
                              continuous_columns=['age', 'hospital_stay'])
    # print(json.dumps(stats, indent=True))
    stats.to_csv(r"data\table.csv", index=False)
