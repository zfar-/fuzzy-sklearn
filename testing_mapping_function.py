from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import sys
from operator import methodcaller
from sklearn.model_selection import train_test_split


class fuzzy_sklearn (object):
    def __init__(self,train_data,test_data):
        self.X_train,self.X_test,self.y_train, self.y_test = train_test_split(train_data)
        self.data = train_data
        self.test_data = test_data
        print('received')

    def main_training_function (self):
        regressor_functions = [self.decision_tree_classifier,self.logistic_regression]
            #, random_forest_classifier, xgboost_classifier,naive_bayes_classifier]

            #value = list(map(lambda x: x(i), classifier_functions))

            #list(map(lambda  x:x , classifier_functions))

        map(methodcaller('__call__', self.train_data), regressor_functions)

    def train_test_split (self,data) :
        x_data = data.iloc[:,:-1]
        y_data = data.iloc[:,-1]
        return train_test_split(x_data, y_data,
                         test_size=0.33, random_state=42)

    def dict_population (self,accuracy ):
        #,                         confusion_matrix ,
        #                 classification_report ) :
        return {'accuracy':accuracy}#, 'confusion_matrix':confusion_matrix , 'classification_report':classification_report }
        #return_dict.udpate({})


    def logistic_regression (self,*args) :

        from sklearn.linear_model import LogisticRegression
        #X_train, X_test, y_train, y_test = train_test_split(data)

        clf = LogisticRegression(random_state=0).fit(self.X_train ,self.y_train)
        y_pred = clf.predict(self.X_test)

        #target_names = ['0','1']

        return  self.dict_population(accuracy= clf.score(self.X_test,self.y_test))
            #,           confusion_matrix= confusion_matrix(y_test, y_pred) ,
            #            classification_report= classification_report(y_test, y_pred, target_names=target_names)
            #)
        #return_dict.update({ 'lg accuracy':clf.score(X_test,y_test)})

        #return_dict.update({'lg confusion_matrix':confusion_matrix(y_test, y_pred)})
        #return_dict.update({'lg classification_report': classification_report(y_test, y_pred, target_names=target_names)})


    def decision_tree_classifier(self,*args):
        from sklearn.tree import DecisionTreeRegressor

        #X_train, X_test, y_train, y_test = train_test_split(data)


        clf = DecisionTreeRegressor(max_depth=50)
       #return_dict = {}
        y_pred = clf.predict(self.X_test)

        #target_names = {'0','1'}
        #return_dict.update({'rf accuracy': metrics.accuracy_score(y_test, y_pred)})
        #return_dict.update({'rf confusion_matrix': confusion_matrix(y_test, y_pred)})
        #return_dict.update({'rf classification_report': classification_report(y_test, y_pred, target_names=target_names)})
        return self.dict_population(accuracy=clf.score(self.X_test,self.y_test))
#                               confusion_matrix=confusion_matrix(self.y_test, y_pred),
#                               classification_report=classification_report(self.y_test, y_pred, target_names=target_names)
#                               )

        #return return_dict

    def random_forest_regressor(self,data):

        from sklearn.ensemble import RandomForestRegressor

        #return_dict = {}
        #X_train, X_test, y_train, y_test = train_test_split(data)

        clf = RandomForestRegressor(max_depth=2)
        clf.fit(self.X_train,self.y_train)
        y_pred = clf.predict(self.X_test)

        target_names=['0','1']
        return self.dict_population(accuracy=metrics.accuracy_score(self.y_test, y_pred),
                               confusion_matrix=confusion_matrix(self.y_test, y_pred),
                               classification_report=classification_report(self.y_test, y_pred, target_names=target_names)
                               )

    def xgboost_classifier (self,data):
        from sklearn.ensemble import RandomForestClassifier

        X_train, X_test, y_train, y_test = train_test_split(data)

        return

    def naive_bayes_classifier(self,data):
        from sklearn.naive_bayes import GaussianNB
        X_train, X_test, y_train, y_test = train_test_split(data)

        #return_dict = {}
        target_names = ['0', '1']
        gnb = GaussianNB()
        y_pred_gnb = gnb.fit(X_train, y_train).predict(X_test)

        return self.dict_population(accuracy=metrics.accuracy_score(y_test, y_pred_gnb),
                               confusion_matrix=confusion_matrix(y_test, y_pred_gnb),
                               classification_report=classification_report(y_test, y_pred_gnb, target_names=target_names)
                               )




'''
if __name__ == '__main__':

    Receiving the data as an argument  
    
    #parser = argparse.ArgumentParser()



    for i in range (1,10,2):
        value = list(map(lambda x : x(i), classifier_functions))
        print(value)

'''

