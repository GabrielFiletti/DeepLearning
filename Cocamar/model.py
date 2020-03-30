import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use(['seaborn', 'ggplot', 'seaborn-white'])
from IPython.core.display import display
import scipy, time, sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, cross_val_predict 
from sklearn.linear_model import LinearRegression as lr, SGDRegressor as sgdr
from sklearn.linear_model import Ridge as rr, RidgeCV as rcvr, Lasso as lassor, LassoCV as lassocvr
from sklearn.neighbors import KNeighborsRegressor as knnr
from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
import random


class Model:
    
    def __init__(self, X, y,  width=100, debug=False):
        self.X = X
        self.y = y
        self.reg = None
        self.testy = None
        self.pred = None
        self.debug = debug
        self.report_width = width
        self.graph_width = 1.3*width//10
        if not self.debug:
            import warnings
            warnings.filterwarnings('ignore')

    def checkDtypes(self, df):
        cols_resume = []
        for col in df.columns:
            coltype = str(df[col].dtype)
            cols_resume.append('%s (%s)' % (col, coltype))
        return cols_resume

    def describe(self, printt=True):

        Xdtypes = self.checkDtypes(self.X)
        ydtypes = self.checkDtypes(self.y)
        if printt:
            print('\nDATA OVERVIEW')
            print(' '*4,'Data shape\n',' '*6,'X: ',self.X.shape,'\n',' '*6,'y: ',self.y.shape)
            print(' '*4,'Columns info\n',' '*6,'X: ',', '.join(Xdtypes),'\n',' '*6,'y: ',', '.join(ydtypes),'\n')

    def redux(self, pca_variance, printt=True, graph=False):
        pca = PCA(n_components = pca_variance)
        pca.fit(self.X)
        before_pca = self.X
        self.X = pca.transform(self.X)

        if printt:
            print('DIMENSIONALITY REDUCTION APPLYING PCA')
            print(' '*4,'Before PCA: ', before_pca.shape)
            print(' '*4,'After PCA: ', self.X.shape,'\n')

        if graph:
            pca = PCA().fit(before_pca)
            plt.figure()
            plt.plot(np.cumsum(pca.explained_variance_ratio_), color='g')
            plt.xlabel('Number of components')
            plt.ylabel('Variance (%)')
            plt.title('Cumulative explained variance')
            plt.show()

    def regression(self, folds=10, printt=True, graph=False):
        size = self.graph_width
        X = self.X
        y = self.y
        safra_range = list(range(len(X.safra.unique())))
        
        models = {}
        models["Linear regressor"]                  = lr()
        models["Lasso CV regressor"]                = lassocvr()
        models["Ridge CV regressor"]                = rcvr()
        models["K nearest neighbors regressor K2u"] = knnr(n_neighbors=2, weights='uniform')
        models["K nearest neighbors regressor K2d"] = knnr(n_neighbors=2, weights='distance')
        models["K nearest neighbors regressor K5"]  = knnr(n_neighbors=5)
        models["K nearest neighbors regressor K10"] = knnr(n_neighbors=10)
        models["Decision tree regressor"]           = dtr()
        models["Decision tree regressor D3"]        = dtr(max_depth=3)
        models["Random forest regressor"]           = rfr()



        report = {"Model":[], "Score (avg)":[], "Score (std)":[], "Elapsed Time(s)":[]}
        for model_name in models:

            score_list = []
            time_list = []
            for i in range(folds):
                rand_ind = random.sample(safra_range,4)
                testX = X[X.safra.isin(rand_ind)]
                testy = y[y.index.isin(testX.index)]
                trainX = X[~X.safra.isin(rand_ind)]
                trainy = y[y.index.isin(trainX.index)]

                start = time.time() 
                model = models[model_name].fit(trainX, trainy)
                score_list.append(model.score(testX, testy))
                time_list.append(time.time()-start)
            
            report["Score (avg)"].append(np.mean(score_list))
            report["Score (std)"].append(np.std(score_list))
            report["Model"].append(model_name)
            report["Elapsed Time(s)"].append(np.mean(time_list))

        report = pd.DataFrame.from_dict(report)
        report.sort_values(by='Score (avg)', inplace=True)
        report.reset_index(inplace=True, drop=True)
        best = report[-1:].values.tolist()[0]
        self.reg = best

        if printt:
            print('REGRESSION RESULTS')
            print('     Best regression method: ', best[0])
            print('     Average score(R2): ', best[1])
            print('     Standard Deviation: ', best[2])
            print('     Elapsed Time(s): ', best[3], '\n')
            #display(report)
            
        if graph:
            model = models[best[0]].fit(trainX, trainy)
            self.pred = model.predict(testX)
            self.testy = testy

            fig, ax = plt.subplots()
            text = 'R2='+str(np.round(best[1],2))
            ax.scatter(testy, self.pred, color='g')
            ax.set_xlabel("True values")
            ax.set_ylabel("Predictions")
            ax.text(0.05, 0.95 , text, transform = ax.transAxes, verticalalignment= 'top', bbox={'boxstyle':'square','facecolor':'none','edgecolor':'black'})
            plt.show()


