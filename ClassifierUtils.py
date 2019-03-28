from pathlib import Path

import JsonLog
import itertools
from multiprocessing import Pool
import multiprocessing

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt




# function that saves and displays features importance list
def makeImportanceList(pos_files, list_of_neg_files):
    f0_pos = pos_files[0]
    f0_neg = corespondeing_neg_file(f0_pos, list_of_neg_files)
    f1_pos = pos_files[1]
    f1_neg = corespondeing_neg_file(f1_pos, list_of_neg_files)

    col_to_drop = ['Unnamed: 0', 'Source', 'Organism', 'microRNA_name', 'miRNA sequence',
                   'target sequence', 'number of reads', 'mRNA_name', 'mRNA_start',
                   'mRNA_end', 'full_mrna', 'site_start']

    if f0_pos == f1_pos:  # one organism. normal case.
        pos = pd.read_csv(f0_pos)
        neg = pd.read_csv(f0_neg)
        X = pd.concat([pos, neg])
        X.reset_index(drop=True, inplace=True)
        X.drop(col_to_drop, axis=1, inplace=True)
        for c in X.columns:
            if c.find("Unnamed") != -1:
                print(" ***************** delete")

                X.drop([c], axis=1, inplace=True)

        y_pos = pd.DataFrame(np.ones((pos.shape[0], 1)))
        y_neg = pd.DataFrame(np.zeros((neg.shape[0], 1)))
        Y = pd.concat([y_pos, y_neg])
        Y.reset_index(drop=True, inplace=True)

    model = ExtraTreesClassifier()
    model.fit(X, Y)

    # display top 20 top features
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nlargest(10).plot(kind='barh')
    plt.show()



# function that performes features selection
def ReduceFeatures(X,Y, featureSelection):
    chi2_selector = SelectKBest(f_regression, k=150)
    X_kbest = chi2_selector.fit_transform(X, Y)
    return X_kbest



# function that find the corresponding negative data file, for a positive data one
def corespondeing_neg_file (pos_file, l):
    term_to_find = str(pos_file.stem).split("_2019")[0]
    term_to_find = term_to_find.replace("pos", "neg")
    term_to_find = term_to_find.replace("_Data","")
    term_to_find = "*{}*".format(term_to_find)
    for x in l:
        if x.match(term_to_find):
            return x
    raise Exception ("Don't find the corespondeing_neg_file {}".format(pos_file))




# make train and test groups
def train_test_prepare(featureSelection,pos_files, list_of_neg_files, test_size=0.2, r_state=42):
    f0_pos = pos_files[0]
    f0_neg = corespondeing_neg_file(f0_pos, list_of_neg_files)
    f1_pos = pos_files[1]
    f1_neg = corespondeing_neg_file(f1_pos, list_of_neg_files)

    col_to_drop = ['Unnamed: 0', 'Source', 'Organism', 'microRNA_name', 'miRNA sequence',
                   'target sequence', 'number of reads', 'mRNA_name', 'mRNA_start',
                   'mRNA_end', 'full_mrna','site_start']

    if f0_pos == f1_pos: # one organism. normal case.
        pos = pd.read_csv(f0_pos)
        neg = pd.read_csv(f0_neg)
        X = pd.concat([pos, neg])
        X.reset_index(drop=True, inplace=True)
        X.drop(col_to_drop, axis=1, inplace=True)
        for c in X.columns:
            if c.find ("Unnamed")!=-1:
                print (" ***************** delete")

                X.drop([c], axis=1, inplace=True)



        y_pos = pd.DataFrame(np.ones((pos.shape[0], 1)))
        y_neg = pd.DataFrame(np.zeros((neg.shape[0], 1)))
        Y = pd.concat([y_pos, y_neg])
        Y.reset_index(drop=True, inplace=True)

        # check if to do feature selection:
        if featureSelection!='no':
            X =ReduceFeatures(X, Y, featureSelection)

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size ,random_state=r_state)
        return  X_train, X_test, y_train.values.ravel(), y_test.values.ravel()

    else: # train is f0 and test is f1 (each is done on a different organism)
        pos_train = pd.read_csv(f0_pos)
        neg_train = pd.read_csv(f0_neg)
        pos_test = pd.read_csv(f1_pos)
        neg_test = pd.read_csv(f1_neg)

        X_train = pd.concat([pos_train, neg_train])
        X_train.reset_index(drop=True, inplace=True)
        X_train.drop(col_to_drop, axis=1, inplace=True)

        y_pos = pd.DataFrame(np.ones((pos_train.shape[0], 1)))
        y_neg = pd.DataFrame(np.zeros((neg_train.shape[0], 1)))
        y_train = pd.concat([y_pos, y_neg])
        y_train.reset_index(drop=True, inplace=True)

        X_test = pd.concat([pos_test, neg_test])
        X_test.reset_index(drop=True, inplace=True)
        X_test.drop(col_to_drop, axis=1, inplace=True)

        y_pos = pd.DataFrame(np.ones((pos_test.shape[0], 1)))
        y_neg = pd.DataFrame(np.zeros((neg_test.shape[0], 1)))
        y_test = pd.concat([y_pos, y_neg])
        y_test.reset_index(drop=True, inplace=True)

        for c in X_test.columns:
            if c.find ("Unnamed")!=-1:
                print (" ***************** delete")
                X_test.drop([c], axis=1, inplace=True)

        for c in X_train.columns:
            if c.find ("Unnamed")!=-1:
                print (" ***************** delete")

                X_train.drop([c], axis=1, inplace=True)

        return  X_train, X_test, y_train.values.ravel(), y_test.values.ravel()


# run the machine learning algorithm
def model_run(training_config, train_X, test_x, train_y, test_y, scoring = "accuracy"):
    best_est = []
    exp_to_run = training_config.keys()
    results = {'name': [], 'f1': [], 'accuracy': []}
    for name in exp_to_run:
        conf = training_config[name]
        clf = conf['clf']
        parameters = conf['parameters']
        n_jobs = conf.get('n_jobs', 1)

        print('=' * 20)
        print('Starting training:', name)
        grid_obj = GridSearchCV(clf, parameters, scoring=scoring, cv=4, n_jobs=n_jobs, verbose=3)

        print('Number of Features:', train_X.columns.shape[0])
        grid_obj = grid_obj.fit(train_X, train_y)
        best_clf = grid_obj.best_estimator_
        best_est.append(best_clf)

        print('Best classifier:', repr(best_clf))
        model = best_clf.fit(train_X, train_y)
        pred_y = model.predict(test_x)

        f1 = f1_score(test_y, pred_y)
        acc = accuracy_score(test_y, pred_y)
        results['name'].append(name)
        results['f1'].append(f1)
        results['accuracy'].append(acc)
        JsonLog.add_to_json("{}_model_params".format(name), model.get_params())

        try:  # if the model is random forest, then make a features importance list
            feature_importances = pd.DataFrame(best_clf.feature_importances_, index=train_X.columns,
                                               columns=['importance'])

            a = feature_importances.sort_values('importance', ascending=False)
            a.to_excel("featuresList_human_Mapping.xlsx")
            print (a.head(10))
            JsonLog.add_to_json("{}_feature_importances".format(name), a.to_dict())

        except Exception:
            print ("This model has no attribute 'feature_importances_")

        print (results)
    JsonLog.add_to_json("results", results)

    return best_est


def main():
    log_dir = Path("d:\\documents\\users\\sheinbey\\Downloads\\Features\\to run pairs\\human_Mapping") # location of saved results logs
    input_pos = Path("d:\\documents\\users\\sheinbey\\Downloads\\Features\\to run pairs\\human_Mapping") # location of positive data
    input_neg = Path("d:\\documents\\users\\sheinbey\\Downloads\\Features\\to run pairs\\human_Mapping\\neg") # location of negative data
    all_neg = list(input_neg.iterdir())   # make list of all negative files
    all_pos = list(input_pos.iterdir()) # make list of all positive files
    all_pos_valid = [x for x in all_pos if x.match("*_pos_valid_seeds_*")] # choose all valid seeds
    all_pos_vienna = [x for x in all_pos_valid if x.match("*vienna*")]
    #all_pos_miranda = [x for x in all_pos_valid if x.match("*miranda*")]

    vienna_all_options = list(itertools.product(all_pos_vienna, all_pos_vienna))

    training_config = {
       'rf': {
           'clf': RandomForestClassifier(),
           'parameters': {
               'n_estimators': [10, 50, 200, 500],
               'criterion': ['gini', 'entropy'],
               'max_depth': [2, 4, 10, 20],
               'min_samples_leaf': [2, 3],
           },
           'n_jobs': 4,
           'one_hot': False
       }
       # 'rbf Kernel': {
       #            'clf': SVC(),
       #            'parameters': {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-4, 1e-5],
       #              'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
       #        }
       #  'logit': {
       #               'clf': LogisticRegression(),
       #               'parameters': {
       #                   'penalty': ['l1', 'l2'],
       #              'C': list(np.arange(0.5, 8.0, 0.1))}
       #
       #  }
    }
    dm="vienna" # duplex method

    # go through all data sets pairs (For different organisms combinations)
    for files_pair in vienna_all_options:
        source =[]
        for f in files_pair:
            tmp = str(f.stem).split("_2019")[0]
            source.append(tmp.split("_Data")[0])

        json_name ="{}____{}.json".format(source[0], source[1])
        print (json_name)

        JsonLog.set_filename(Path(log_dir) / json_name)
        JsonLog.add_to_json("source",source[0])
        JsonLog.add_to_json("test",source[1])
        JsonLog.add_to_json("duplex_method", dm)


        X_train, X_test, y_train, y_test = train_test_prepare('no',files_pair, all_neg, test_size=0.2, r_state=42)
        JsonLog.add_to_json("train size", X_train.shape[0])
        JsonLog.add_to_json("test size", X_test.shape[0])
        best_est = model_run(training_config, X_train, X_test, y_train, y_test, scoring="accuracy")
        #makeImportanceList(files_pair,all_neg)

def intersectList(path,list1,list2):
    dfHuman = pd.read_excel(path+'\\'+list1+'.xlsx')
    dfMouse = pd.read_excel(path+'\\'+list2+'.xlsx')
    featuresListHuman = dfHuman.index.tolist()
    print(len(featuresListHuman))
    featuresListMouse = dfMouse.index.tolist()
    print(len(featuresListMouse))
    a= featuresListHuman[:50]
    b=featuresListMouse[:50]
    print(len(a))
    result = intersection(a,b)

    # save to excel
    df=pd.DataFrame()
    df['common features'] = result
    df.to_excel('common_features'+'\\'+list1+''+list2+'.xlsx')

    print (len(result))

def commonFeatures(path):
    input=Path(path)
    all_lists = list(input.iterdir())
    all_options = list(itertools.product(all_lists, all_lists))
    for files_pair in all_options:
        source =[]
        for f in files_pair:
            tmp = str(f.stem).split("_2019")[0]
            source.append(tmp.split("_Data")[0])

        list1 = source[0]
        list2 = source[1]
        if (list1 != list2):
            intersectList(path, list1, list2)



def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    print (lst3)
    print (len(lst3))
    return lst3

def intersection_ExcelFiles(path):
    input = Path(path)
    all_lists = list(input.iterdir())
    df1 = pd.read_excel(all_lists[0])
    df2 = pd.read_excel(all_lists[1])
    list1 = df1.index.tolist()
    list2 = df1.index.tolist()
    # list2 = df2['Feature name'].tolist() --- used when there is a column name
    list1=list1[:50]
    list2 = list2[:50]
    intersection(list1,list2)

if __name__ == "__main__":
    # main()
    # intersectList()
    # commonFeatures('C:\\Users\\sheinbey\\PycharmProjects\\mirnaProject\\lists')
    intersection_ExcelFiles('C:\\Users\\sheinbey\\PycharmProjects\\mirnaProject\\lists\\test')
