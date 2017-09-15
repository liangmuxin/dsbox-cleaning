import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def _discretize_by_width(col, num_bins, labels):
    maxvalue = col.max()
    minvalue = col.min()
    width = float((maxvalue-minvalue))/num_bins
    bins = [minvalue + x*width for x in range(num_bins)]+[maxvalue]
    if labels:
        if len(labels)!=num_bins:
            raise ValueError('Length of assigned labels not consistent with num_bins!')
        else:
            group_names = labels
    else:
        group_names = range(num_bins)
    return pd.cut(col, bins,labels=group_names, include_lowest=True)


def _discretize_by_frequency(col, num_bins, labels):
    percent = 1.0/num_bins
    bins = sorted(list(set(col.quantile([x*percent for x in range(num_bins+1)]))))
    if len(bins)-1 < num_bins:
        num_bins = len(bins)-1
        print('...Only %d bins (unbalanced) generated due to overlapping percentile boundaries.'%num_bins)
    if labels:
        if len(labels)!=num_bins:
            raise ValueError('Length of assigned labels not consistent with num_bins!')
        else:
            group_names = labels
    else:
        group_names = range(num_bins)
    return pd.cut(col, bins,labels=group_names, include_lowest=True)


def _discretize_by_kmeans(col, num_bins, random_state):
    nan_idx = col[col.isnull()].index
    kmeans = KMeans(n_clusters=num_bins, random_state=random_state)
    kmeans = kmeans.fit(col.dropna().values.T.reshape(-1, 1))
    group = kmeans.labels_
    if col.isnull().sum() > 0:
        group = group.astype(float)
        for idx in nan_idx:
            group = np.insert(group,idx,np.nan)
    return pd.Series(group)


def _discretize_by_gmm(col, num_bins, random_state):
    nan_idx = col[col.isnull()].index
    gmm = GaussianMixture(n_components=num_bins,covariance_type='full',random_state=random_state)
    gmm = gmm.fit(X=np.expand_dims(col.dropna(), 1))
    if col.isnull().sum() == 0:
        group = gmm.predict(X=np.expand_dims(col, 1))
    else:
        group = gmm.predict(X=np.expand_dims(col.dropna(), 1)).astype(float)
        for idx in nan_idx:
            group = np.insert(group,idx,np.nan)
    return pd.Series(group)


def discretize(col, num_bins=10, by='width', labels = None, random_state=0):
    if col.dropna().sum() == 0:
        raise ValueError('Empty column!')
    if by == 'width':
        return _discretize_by_width(col, num_bins, labels)

    elif by == 'frequency':
        return _discretize_by_frequency(col, num_bins, labels)

    elif by == 'kmeans':
        if labels:
            print('...Applying kmeans clustering, so user-defined labels are ignored.')
        return _discretize_by_kmeans(col, num_bins, random_state)

    elif by == 'gmm':
        if labels:
            print('...Applying gmm clustering, so user-defined labels are ignored.')
        return _discretize_by_gmm(col, num_bins, random_state)

    else:
        raise ValueError('...Invalid by (binning method) parameter %s'%by)



class Discretizer(object):
    """
    """

    def __repr__(self):
        return "%s(%r)" % ('DSBoxDiscretizer', self.__dict__)


    def __init__(self, by='gmm', num_bins=10, random_state=0):
        self.by = by
        self.random_state = random_state
        self.num_bins = num_bins
        self.bins = {}
        self.model = {}


    def __discretize_by_width(self, col):
        maxvalue = col.max()
        minvalue = col.min()
        width = (maxvalue-minvalue)/self.num_bins
        abin = [minvalue + x*width for x in range(self.num_bins)]+[maxvalue]
        self.bins[col.name] = abin


    def __discretize_by_frequency(self, col):
        percent = 1.0/self.num_bins
        abin = sorted(list(set(col.quantile([x*percent for x in range(self.num_bins+1)]))))
        self.bins[cn] = abin


    def __discretize_by_kmeans(self, col):
        kmeans = KMeans(n_clusters=self.num_bins, random_state=self.random_state)
        kmeans = kmeans.fit(col.dropna().values.T.reshape(-1, 1))
        centers = kmeans.cluster_centers_
        centers.sort(axis=0)
        abin = np.vstack([centers.T[0][1:], centers.T[0][:-1]]).mean(axis=0).tolist()
        self.bins[col.name] = [col.min()]+abin+[col.max()]


    def __discretize_by_gmm(self,col):
        gmm = GaussianMixture(n_components=self.num_bins,random_state=self.random_state)
        gmm = gmm.fit(X=np.expand_dims(col.dropna(), 1))
        self.model[col.name] = gmm
        self.bins[col.name] = None

    def fit(self, data, label=None):
        """
        """
        if not isinstance(data, pd.DataFrame):
            data = pd.read_csv(data)
        data_copy = data.copy()
        
        for cn in data_copy:
            col = data_copy[cn]
            if col.dropna().count() == 0:
                pass
            elif col.dtype.kind in np.typecodes['AllFloat']:
                if self.by == "width":
                    self.__discretize_by_width(col)
        
                elif self.by == "frequency":
                    self.__discretize_by_frequency(col)
        
                elif self.by == "kmeans":
                    self.__discretize_by_kmeans(col)

                elif self.by == "gmm":
                    self.__discretize_by_gmm(col)

                else:
                    raise ValueError('(fit)...Invalid by (binning method) parameter %s'%self.by)


    def transform(self, data, label=None):
        ""
        ""
        if not isinstance(data, pd.DataFrame):
            data = pd.read_csv(data)
        data_copy = data.copy()
        
        result = pd.DataFrame()
        
        if self.by == 'gmm':
            for cn in data:
                col = data[cn]
                if cn in list(self.bins.keys()):
                    nan_idx = col[col.isnull()].index
                    new_col = self.model[cn].predict(X=np.expand_dims(col.dropna(),1)).astype(float)
                    if col.isnull().count() > 0:
                        for idx in nan_idx:
                            new_col = np.insert(new_col, idx,np.nan)
                    new_col = pd.Series(new_col)

                    tmp = pd.DataFrame()
                    tmp['orig'] = col.sort_values()
                    tmp['aft'] = new_col
                    tmp_dict = {}
                    for i in range(self.num_bins):
                        tmp_dict[i] = tmp['aft'].unique()[i]
                    
                    new_col = new_col.map(tmp_dict)
                    result[cn] = new_col
                else:
                    result[cn] = col.copy()
            
            return result

        elif self.by in ['width', 'frequency', 'kmeans']:
            for cn in data:
                col = data[cn]
                if cn in list(self.bins.keys()):
                    abin = self.bins[col.name]
                    group = range(len(abin)-1)
                    result[cn] = pd.cut(col, abin, labels=group, include_lowest=True)
                else:
                    result[cn] = col.copy()
        
            return result
        
        else:
            raise ValueError('(transform)...Invalid by (binning method) parameter %s'%self.by)
