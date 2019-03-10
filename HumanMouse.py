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






def corespondeing_neg_file (pos_file, l):
    term_to_find = str(pos_file.stem).split("_2019")[0]
    term_to_find = term_to_find.replace("pos", "neg")
    term_to_find = term_to_fcorespondeing_neg_fileind.replace("_Data","")
    term_to_find = "*{}*".format(term_to_find)
    for x in l:
        if x.match(term_to_find):
            return x
    raise Exception ("Don't find the corespondeing_neg_file {}".format(pos_file))





def train_test_prepare(human_pos_file, human_neg_file, mouse_pos_file, test_size=0.2, r_state=42):
    def remove_unnecessary_columns (df):
        col_to_drop = ['Unnamed: 0', 'Source', 'Organism', 'microRNA_name', 'miRNA sequence',
                       'target sequence', 'number of reads', 'mRNA_name', 'mRNA_start',
                       'mRNA_end', 'full_mrna','site_start']
        df.drop(col_to_drop, axis=1, inplace=True)
        for c in df.columns:
            if c.find("Unnamed") != -1:
                print (" ***************** delete")

                df.drop([c], axis=1, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df



    human_pos = pd.read_csv(human_pos_file)
    human_neg = pd.read_csv(human_neg_file)
    mouse_pos = pd.read_csv(mouse_pos_file)
  #  mouse_neg = pd.read_csv(mouse_neg_file)

    human_pos['organism'] = 1000
    human_neg['organism'] = 1000
    mouse_pos['organism'] = 2000
   # mouse_neg['organism'] = "mouse"

    X_human = pd.concat([human_pos, human_neg])
    X_human = remove_unnecessary_columns(X_human)
    X_mouse_pos = remove_unnecessary_columns(mouse_pos)

    y_pos = pd.DataFrame(np.ones((human_pos.shape[0], 1)))
    y_neg = pd.DataFrame(np.zeros((human_neg.shape[0], 1)))
    y_mouse_pos = pd.DataFrame(np.ones((mouse_pos.shape[0], 1)))

    Y_human = pd.concat([y_pos, y_neg])
    Y_human.reset_index(drop=True, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(X_human, Y_human, test_size=test_size ,random_state=r_state)
    X_train_mouse = pd.concat([X_train, X_mouse_pos])
    X_train_mouse.reset_index(drop=True, inplace=True)
    y_train_mouse = pd.concat([y_train, y_mouse_pos])
    y_train_mouse.reset_index(drop=True, inplace=True)

    human_only = (X_train, X_test, y_train.values.ravel(), y_test.values.ravel())
    human_with_mouse = (X_train_mouse, X_test, y_train_mouse.values.ravel(), y_test.values.ravel())
    return  human_only, human_with_mouse


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

        try:
            feature_importances = pd.DataFrame(best_clf.feature_importances_, index=train_X.columns,
                                               columns=['importance'])
            a = feature_importances.sort_values('importance', ascending=False)
            print (a.head(10))
            JsonLog.add_to_json("{}_feature_importances".format(name), a.to_dict())


        except Exception:
            print ("This model has no attribute 'feature_importances_")

        print (results)
    JsonLog.add_to_json("results", results)

    return best_est


def main():
    #  sudo /sbin/service sshd start
    log_dir = Path("Data/Results")

    input_pos = Path("Data/Features/CSV")
    input_neg = Path("Data/Features/CSV/Neg")

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
        # 'rf': {
        #     'clf': RandomForestClassifier(),
        #     'parameters': {
        #         'n_estimators': [ 500],
        #         'criterion': [ 'entropy'],
        #         'max_depth': [20],
        #         'min_samples_leaf': [ 3],
        #     },
        #     'n_jobs': 4,
        #     'one_hot': False
        # }

    }

    human_pos_f = input_pos / "human_Mapping_the_Human_miRNA_Data_vienna_pos_valid_seeds_20190130-015129.csv"
    human_neg_f = input_neg / "human_Mapping_the_Human_miRNA_vienna_neg_valid_seeds_20190201-195329.csv"
    mouse_pos_f = input_pos / "mouse_Unambiguous_Identification_Data_vienna_pos_valid_seeds_20190129-235706.csv"

    json_name ="human_without_mouse_support.json"
    JsonLog.set_filename(Path(log_dir) / json_name)
    JsonLog.add_to_json("source", "human_mouse_mouse")
    JsonLog.add_to_json("test","human")
    JsonLog.add_to_json("duplex_method", "vienna")


    human_only, human_with_mouse = train_test_prepare(human_pos_f, human_neg_f, mouse_pos_f, test_size=0.2, r_state=42)
    X_train, X_test, y_train, y_test =  human_only
    JsonLog.add_to_json("train size", X_train.shape[0])
    JsonLog.add_to_json("test size", X_test.shape[0])
    best_est = model_run(training_config, X_train, X_test, y_train, y_test, scoring="accuracy")

    # valid_seeds_files = [x for x in all_files if x.match("*_pos_valid_seeds_*")]
    # files_with_params = [(x, x.name.split("_")[0], x.name.split("_Data_")[1].split("_")[0]) for x in valid_seeds_files]
    #
    # # output_dir = Path("Data/Features/CSV")
    # # log_dir = Path("Data/Features/Logs")
    #
    #
    # duplex_method = CONFIG['duplex_method']
    # f = list(input_dir.iterdir())
    # all_work = list(itertools.product(f, duplex_method))
    #
    # p = Pool(CONFIG['max_process'])
    # p.map(worker, all_work)
    # p.close()
    # p.join()
    #


if __name__ == "__main__":
    main()

