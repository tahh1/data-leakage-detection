#!/usr/bin/env python
# coding: utf-8

# # Imports

# In this cell we are importing all relevant packages for our project

# In[1]:


# connections and OS
import pandas as pd
import seaborn as sns
import os
import sqlite3
import csv

#utils (Pandas,numpy,tqdm)
import numpy as np
import pandas as pd
from tqdm import tqdm

#visualize 
import seaborn as sns

#preprocessing, metrices and splits 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler,MinMaxScaler

#ML models:
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression, Ridge


#tensorflow layer, callbacks and layers
import tensorflow.keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Concatenate, Input
from tensorflow.keras.callbacks import ModelCheckpoint, Callback


# # Constants and params

# Change directory to project directory within the department cluster (SLURM) and define to constant variables
# 

# In[2]:


#change directory to project directory within the department cluster (SLURM)
os.chdir('../../storage/users/AmitAdirProject/Data/sqlite/')
CHANNELS = ["AGP","DNA","ER","Mito","RNA"]


# # Preprocessing functions

# In[3]:


def scale_data(df):
    """
    This function is scaling the data using two methods: STD and MinMax
    df: dataFrame     
    return: two scaled dataframes, the first one according to StandardScaler and the second according to MinMaxScaler
    """
    std_scalar = StandardScaler()
    minMax_scalar = MinMaxScaler()
    
    std_df = pd.DataFrame(std_scalar.fit_transform(df),columns=df.columns)
    minmax_df = pd.DataFrame(minMax_scalar.fit_transform(df),columns=df.columns)    
    std_df.fillna(0,inplace=True)
    minmax_df.fillna(0,inplace=True)
    
    return std_df,minmax_df


# In[17]:


def append_cells_labels(filename):
    """    
    This function is responsible for creating label (Mock/Tretament) for cell resultion and create a corresponding csv file
    filename: the full file path    
    """
    cnx2 = sqlite3.connect(filename)
    df_image = pd.read_sql_query("SELECT ImageNumber,Image_Metadata_Well FROM Image", cnx2)
    table_name = filename[44:-7]
    df_well = pd.read_csv("//storage//users//AmitAdirProject//Data//mean_well_profiles//"+table_name+".csv")
    df_well = df_well[["Metadata_Well","Metadata_ASSAY_WELL_ROLE"]]
    print('Done reading Image sql')
    
    dict_treatment = {} 
    for i, row in df_image.iterrows():
        label = df_well.loc[df_well['Metadata_Well'] == row['Image_Metadata_Well'], 'Metadata_ASSAY_WELL_ROLE'].iloc[0]
        dict_treatment[row['ImageNumber']] = label
    
    
    dict_channel = {}
    # Create your connection.
    cnx = sqlite3.connect(filename)
    #df is the cell dataframe
    df = pd.read_sql_query("SELECT * FROM Cells", cnx)
    df['label'] = df['ImageNumber'].map(dict_treatment)
    df.to_csv("/storage/users/AmitAdirProject/Data/sqlite/"+table_name+".csv")
    print(("Done with " + table_name))


# Now, we will generate a csv for each plate with the corresponding mock/treated data

# In[18]:


path_plates = "/storage/users/AmitAdirProject/Data/sqlite/"
for plate in tqdm(os.listdir(path_plates)):
    if plate.endswith(".sqlite"):
        append_cells_labels(path_plates+"/"+plate)


# In[19]:


def split_channels_x_and_y(filename, task_channel):
    """
    This function is responsible for splitting five channels into four channels as train and the remaining channel to test
    filename: file path to the cell table from a single plate
    task_channel: the current channel that we aim to predict
    
    Notably: In order to avoid leakage we drop all 'correlation features
    return: seperated dataframes x_features and y_df. 
            x_features: contains all available features excluding the features related to 'task_channel' we aim to predict
            y_df: contains all available features related to 'task_channel' only
    """
    dict_channel = {}
    
    #df = pd.read_csv(filename+".csv")
    df = pd.read_csv(filename)
    df = df.set_index(['ImageNumber', 'ObjectNumber'])
    df.drop(['TableNumber'],inplace=True,axis=1)
    df.dropna(inplace=True)
    
    # Data Preperation
    general_featuers = df.iloc[:, 0:52]
    general_featuers['label'] = df['label']
    df = df.iloc[:, 52:]
    for channel in CHANNELS:
        dict_channel[channel] = df[[col for col in df.columns if channel in col]]

    ready_channel_features = []
    for feature_name in dict_channel:
        if feature_name != task_channel:
            curr_channel_features = dict_channel[feature_name]
            curr_channel_features = curr_channel_features[[col for col in curr_channel_features.columns if task_channel not in col]]
            ready_channel_features.append(curr_channel_features)
    x_features_df = ready_channel_features[0]    
    for i in range(1, len(ready_channel_features)):
        x_features_df = x_features_df.join(ready_channel_features[i], how='outer', lsuffix='_left',
                                           rsuffix='_right')
    x_features_df = general_featuers.join(x_features_df, how='outer', lsuffix='_left', rsuffix='_right')
    y_df = dict_channel[task_channel]
    corr_cols = [c for c in y_df.columns if 'correlation' not in c.lower()]
    y_df = y_df[corr_cols]
    return x_features_df, y_df


# # Create Models

# In the following three cells we are creating three ML models

# In[28]:


def create_LR(df_train_X, df_train_Y):  
    """
    In this cell we are creating and training a linear regression model        
    df_train_X: contains all available features excluding the features related to 'task_channel' we aim to predict (train)
    df_train_Y: contains all available features related to 'task_channel' only for the train
    
    
    return: trained linear regression model
    """
    LR_model = LinearRegression()    
    LR_model.fit(df_train_X.values,df_train_Y.values)
    return LR_model
    
    


# In[29]:


def create_Ridge(df_train_X, df_train_Y):
    """    
    In this cell we are creating and training a ridge regression model    
    
    
    df_train_X: contains all available features excluding the features related to 'task_channel' we aim to predict (train)
    df_train_Y: contains all available features related to 'task_channel' only for the train    
    
    return: trained linear regression model
    """
    Ridge_model = Ridge()    
    Ridge_model.fit(X=df_train_X.values,y=df_train_Y.values)    
    return Ridge_model


# In[30]:


def create_model_dnn(task_channel,df_train_X, df_train_Y,test_plate):
    """    
    In this cell we are creating and training a multi layer perceptron (we refer to it as deep neural network, DNN) model
    
    task_channel: the current channel that we aim to predict
    df_train_X: contains all available features excluding the features related to 'task_channel' we aim to predict (train)
    df_train_Y: contains all available features related to 'task_channel' only for the train
    channel_task_x: contains all available features excluding the features related to 'task_channel' we aim to predict (test)
    channel_task_y: contains all available features related to 'task_channel' only for the test
    test_plate: the ID of a given plate. This information assist us while printing the results.
    
    return: trained linear regression model
    """
    # Stracture of the network#
    inputs = Input(shape=(df_train_X.shape[1],))
    dense1 = Dense(512,activation = 'relu')(inputs)
    dense2 = Dense(256,activation = 'relu')(dense1)
    dense3 = Dense(128,activation = 'relu')(dense2)    
    dense4 = Dense(100,activation = 'relu')(dense3)
    dense5 = Dense(50,activation = 'relu')(dense4)
    dense6 = Dense(25,activation = 'relu')(dense5)
    dense7 = Dense(10,activation = 'relu')(dense6)
    predictions = Dense(df_train_Y.shape[1],activation='sigmoid')(dense7)
    
    #model compiliation
    model = Model(inputs=inputs,outputs = predictions)
    model.compile(optimizer='adam',loss='mse')
    
    #model training    
    test_plate_number = test_plate[:5]
    filepath = os.path.join('../../Models/',f'{test_plate_number}_{task_channel}.h5')
    my_callbacks = [ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)]
    model.fit(df_train_X,df_train_Y,epochs = 5,batch_size=1024*8,verbose=0,shuffle=True,validation_split=0.2,callbacks=my_callbacks)
    return model
    
    


# In[31]:


#    print_results(test_plate_number, task_channel, "Overall", "DNN", "None", "MSE", str(mean_squared_error(model_pred,channel_task_y)))
def print_results(plate_number, channel, family, model, _type, metric, value):
    """
    This function is creating a csv named: 'results' that contains all of the models’ performance (e.g. MSE) for each plate and each family of attributes
    plate_number: ID of palte
    channel: The channel we aim to predict
    family: features united by their charactheristics (e.g., Granularity, Texture)
    model: the model name
    _type: scaling method (e.g., MinMax Scaler or StandardScaler)
    metric: MSE/MAE
    value: value of the metric error    
    """
    results_path = "/storage/users/AmitAdirProject/Data/Results/"
    files_list = os.listdir(results_path)
    if 'results.csv' not in files_list:
        file1 = open("/storage/users/AmitAdirProject/Data/Results/results.csv","a+")     
        file1.write("Plate,Channel,Family,Model,Type,Metric,Value \n")
        file1.write(plate_number+","+channel+","+family+","+model+","+_type+","+metric+","+value+"\n")
        file1.close()
    else:
        file1 = open("/storage/users/AmitAdirProject/Data/Results/results.csv","a+")
        file1.write(plate_number+","+channel+","+family+","+model+","+_type+","+metric+","+value+"\n")
        file1.close()


# In[32]:


def get_family_MSE(test_plate_number, task_channel, model, _type, df, channel_task_y):    
    """
    This function is calculating the MSE/MAE measures for plates based on different models
    test_plate_number: ID of the examine plate
    task_channel: Channel we aim to predict
    model: model name
    _type: scaling method (e.g., MinMax Scaler or StandardScaler)
    df: prediction of any given ML model which aim to predict the channel_task_y
    channel_task_y: features corresponding to the 'task channel' (channel we aim to predict)    
    """
    Families = {'Granularity':[],
               'Intensity':[],
               'Location':[],
               'RadialDistribution':[],
               'Texture':[]}

    for name in (channel_task_y.columns):
        if '_Granularity' in name:
            Families['Granularity'].append(name)
        elif '_Intensity' in name:
            Families['Intensity'].append(name)
        elif '_Location' in name:
            Families['Location'].append(name)        
        elif '_RadialDistribution' in name:
            Families['RadialDistribution'].append(name)
        elif '_Texture' in name:
            Families['Texture'].append(name)
            
    for key in list(Families.keys()):
        try:            
            print_results(test_plate_number, task_channel, key, model, _type, "MSE", str(mean_squared_error(df[Families[key]],channel_task_y[Families[key]])))
        except:
            if len(Families[key]) == 0:
                print(('empty family {}'.format(key)))
            else:
                print('problem in mse key')
            



# In[33]:


def get_family_MAE(test_plate_number, task_channel, model, _type, df, channel_task_y):
    
    """
    This function is calculating the MSE/MAE measures for plates based on different models
    test_plate_number: ID of the examine plate
    task_channel: Channel we aim to predict
    model: model name
    _type: scaling method (e.g., MinMax Scaler or StandardScaler)
    df: prediction of any given ML model which aim to predict the channel_task_y
    channel_task_y: features corresponding to the 'task channel' (channel we aim to predict)    
    """
    
    Families = {'Granularity':[],
               'Intensity':[],
               'Location':[],
               'RadialDistribution':[],
               'Texture':[]}

    for name in (channel_task_y.columns):
        if '_Granularity' in name:
            Families['Granularity'].append(name)
        elif '_Intensity' in name:
            Families['Intensity'].append(name)
        elif '_Location' in name:
            Families['Location'].append(name)        
        elif '_RadialDistribution' in name:
            Families['RadialDistribution'].append(name)
        elif '_Texture' in name:
            Families['Texture'].append(name)
            
    for key in list(Families.keys()):
        try:            
            print_results(test_plate_number, task_channel, key, model, _type, "MAE", str(mean_absolute_error(df[Families[key]],channel_task_y[Families[key]])))
        except:
            if len(Families[key]) == 0:
                print(('empty family {}'.format(key)))
            else:
                print('problem in mae key')
        


# In[34]:


def main(path,scale_method):
    """
    This is the main function of the preprocessing steps.
    This function will iterate all over the sqlite files and do the following:
    1) prepate train + test files
    2) scale train + test files (x + y values separately)
    3) return: 
        task_channel -> string, reflect the relevant channel for test. For example, 'AGP'
        df_train_X -> DataFrame, (instances,features) for the train set
        df_train_Y -> DataFrame, (instances,labels) for the train set
        channel_task_x -> DataFrame, (instances,features) for the test set
        channel_task_y -> DataFrame, (instances,labels) for the test set
    """
    path_profiles = '/storage/users/AmitAdirProject/Data/mean_well_profiles/'
    csv_files= [_ for _ in os.listdir('/storage/users/AmitAdirProject/Data/sqlite/') if _.endswith(".csv")]    
    for task_channel in tqdm(CHANNELS):        
        # This is the current file that we will predict        
        for test_plate in csv_files:
            print(test_plate)
            if test_plate.endswith(".csv"):                
                channel_task_x, channel_task_y = split_channels_x_and_y(path + test_plate, task_channel)
                print((channel_task_x['label'].unique()))
                
                channel_task_x_mock = channel_task_x[channel_task_x['label']=='mock']
                channel_task_x_treated = channel_task_x[channel_task_x['label']=='treated']
                
                channel_task_y_mock = channel_task_y.loc[channel_task_x_mock.index]
                channel_task_y_treated = channel_task_y.loc[channel_task_x_treated.index]
                
                channel_task_x_mock.drop(['label'],inplace=True,axis=1)
                channel_task_x_treated.drop(['label'],inplace=True,axis=1)
                
                
                std_df_treated_x ,min_max_df_treated_x = scale_data(channel_task_x_treated)
                std_df_treated_y ,min_max_df_treated_y = scale_data(channel_task_y_treated)
                std_df_mock_x ,min_max_df_mock_x = scale_data(channel_task_x_mock)
                std_df_mock_y ,min_max_df_mock_y = scale_data(channel_task_y_mock)

                
        # This is all other files X input
            list_x_df = []
            list_y_df = []
            
            
            for train_plate in tqdm(csv_files):
                if train_plate!=test_plate:
                    if train_plate.endswith(".csv"):
                        curr_x, curr_y = split_channels_x_and_y(path + train_plate, task_channel)
                        curr_x = curr_x[curr_x['label']=='mock']
                        curr_y = curr_y.loc[curr_x.index]                                              
                        curr_x.drop(['label'],inplace=True,axis=1)
                        
                        list_x_df.append(curr_x)                        
                        list_y_df.append(curr_y)
            
            df_train_X = pd.concat(list_x_df)
            df_train_Y = pd.concat(list_y_df)   
            
             # Scale for training set#
            std_df ,min_max_df = scale_data(df_train_X)            
            std_df_y ,min_max_df_y = scale_data(df_train_Y)
            
            #Scale for testing set - treated#
            std_df_channel_task_treated ,min_max_df_channel_task_treated = scale_data(channel_task_x_treated)
            std_df_y_test_treated ,min_max_df_y_test_treated = scale_data(channel_task_y_treated)
            
            #Scale for testing set - mock#
            std_df_channel_task_mock ,min_max_df_channel_task_mock = scale_data(channel_task_x_mock)
            std_df_y_test_mock ,min_max_df_y_test_mock = scale_data(channel_task_y_mock)   
            
            if scale_method == 'MinMax':
                #train set#
                df_train_X = min_max_df
                df_train_Y = min_max_df_y
                
                #treated #
                df_test_X_treated = min_max_df_channel_task_treated
                df_test_Y_treated = min_max_df_y_test_treated
                
                #mock#                
                df_test_X_mock = min_max_df_channel_task_mock
                df_test_Y_mock = min_max_df_y_test_mock
                
                
            elif scale_method == 'Std':
                #train set#
                df_train_X = std_df
                df_train_Y = std_df_y
                
                #treated #
                df_test_X_treated = std_df_channel_task_treated
                df_test_Y_treated = std_df_y_test_treated
                
                #mock#                
                df_test_X_mock = std_df_channel_task_mock
                df_test_Y_mock = std_df_y_test_mock
                
                
        # Model Creation - AVG MSE for each model:
            print(test_plate)
            print((task_channel+":"))
            LR_model = create_LR(df_train_X, df_train_Y)
            Ridge_model = create_Ridge(df_train_X, df_train_Y)
            DNN_model = create_model_dnn(task_channel,df_train_X, df_train_Y,test_plate)
#             svr_model = create_SVR(task_channel,df_train_X, df_train_Y, channel_task_x, channel_task_y)
            #return task_channel,df_train_X, df_train_Y, channel_task_x, channel_task_y
    
            print('**************')
            print('LR')
            print('profile_treated:') 
            yhat_lr = pd.DataFrame(LR_model.predict(std_df_treated_x.values),columns=std_df_treated_y.columns)                           
            print(('Linear Reg MSE: {}'.format(mean_squared_error(yhat_lr,std_df_treated_y.values))))  
            print(('Linear Reg MAE: {}'.format(mean_absolute_error(yhat_lr,std_df_treated_y.values))))
            
            print_results(test_plate, task_channel, 'Overall', 'Linear Regression', 'Treated', 'MSE', str(mean_squared_error(yhat_lr,std_df_treated_y.values)))
            print_results(test_plate, task_channel, 'Overall', 'Linear Regression', 'Treated', 'MAE', str(mean_absolute_error(yhat_lr,std_df_treated_y.values)))
            
            get_family_MSE(test_plate, task_channel, "Linear Regression", "Treated", yhat_lr,std_df_treated_y)
            get_family_MAE(test_plate, task_channel, "Linear Regression", "Treated", yhat_lr,std_df_treated_y)
            
            #get_family_MSE(yhat_lr,std_df_treated_y)
            #get_family_MAE(yhat_lr,std_df_treated_y)
            
            print('profile_mock:')            
            yhat_lr = pd.DataFrame(LR_model.predict(std_df_mock_x.values),columns=std_df_mock_y.columns)   
            print(('Linear Reg MSE: {}'.format(mean_squared_error(yhat_lr,std_df_mock_y.values))))  
            print(('Linear Reg MAE: {}'.format(mean_absolute_error(yhat_lr,std_df_mock_y.values))))  
            
            print_results(test_plate, task_channel, 'Overall', 'Linear Regression', 'Mock', 'MSE', str(mean_squared_error(yhat_lr,std_df_mock_y.values)))
            print_results(test_plate, task_channel, 'Overall', 'Linear Regression', 'Mock', 'MAE', str(mean_absolute_error(yhat_lr,std_df_mock_y.values)))            
            #get_family_MSE(yhat_lr,std_df_mock_y)
            #get_family_MAE(yhat_lr,std_df_mock_y)
            get_family_MSE(test_plate, task_channel, "Linear Regression", "Mock", yhat_lr,std_df_mock_y)
            get_family_MAE(test_plate, task_channel, "Linear Regression", "Mock", yhat_lr,std_df_mock_y)
                          
            print('**************')
            
            print('**************')
            print('Ridge')
            print('profile_treated:') 
            yhat_ridge = pd.DataFrame(Ridge_model.predict(std_df_treated_x.values),columns=std_df_treated_y.columns)                           
            print(('Ridge MSE: {}'.format(mean_squared_error(yhat_ridge,std_df_treated_y.values))))  
            print(('Ridge MAE: {}'.format(mean_absolute_error(yhat_ridge,std_df_treated_y.values))))  
                          
            print_results(test_plate, task_channel, 'Overall', 'Ridge', 'Treated', 'MSE', str(mean_squared_error(yhat_ridge,std_df_treated_y.values)))
            print_results(test_plate, task_channel, 'Overall', 'Ridge', 'Treated', 'MAE', str(mean_absolute_error(yhat_ridge,std_df_treated_y.values)))
                    
                          
            get_family_MSE(test_plate, task_channel, "Ridge", "Treated", yhat_ridge,std_df_treated_y)
            get_family_MAE(test_plate, task_channel, "Ridge", "Treated", yhat_ridge,std_df_treated_y)
                          
            #get_family_MSE(yhat_lr,std_df_treated_y)
            #get_family_MAE(yhat_lr,std_df_treated_y)
            
            print('profile_mock:')            
            yhat_ridge = pd.DataFrame(Ridge_model.predict(std_df_mock_x.values),columns=std_df_mock_y.columns)   
            print(('Ridge MSE: {}'.format(mean_squared_error(yhat_ridge,std_df_mock_y.values))))  
            print(('Ridge Reg MAE: {}'.format(mean_absolute_error(yhat_ridge,std_df_mock_y.values))))  
            #get_family_MSE(yhat_ridge,std_df_mock_y)
            #get_family_MAE(yhat_ridge,std_df_mock_y)
            print_results(test_plate, task_channel, 'Overall', 'Ridge', 'Mock', 'MSE', str(mean_squared_error(yhat_ridge,std_df_mock_y.values)))
            print_results(test_plate, task_channel, 'Overall', 'Ridge', 'Mock', 'MAE', str(mean_absolute_error(yhat_ridge,std_df_mock_y.values)))
                    
                          
            get_family_MSE(test_plate, task_channel, "Ridge", "Mock", yhat_ridge,std_df_mock_y)
            get_family_MAE(test_plate, task_channel, "Ridge", "Mock", yhat_ridge,std_df_mock_y)
            print('**************')
            
            print('**************')
            print('DNN')
            print('profile_treated:') 
            yhat_DNN = pd.DataFrame(DNN_model.predict(std_df_treated_x.values),columns=std_df_treated_y.columns)                           
            print(('DNN MSE: {}'.format(mean_squared_error(yhat_DNN,std_df_treated_y.values))))  
            print(('DNN MAE: {}'.format(mean_absolute_error(yhat_DNN,std_df_treated_y.values))))  
            #get_family_MSE(yhat_DNN,std_df_treated_y)
            #get_family_MAE(yhat_DNN,std_df_treated_y)
            print_results(test_plate, task_channel, 'Overall', 'DNN', 'Treated', 'MSE', str(mean_squared_error(yhat_DNN,std_df_treated_y.values)))
            print_results(test_plate, task_channel, 'Overall', 'DNN', 'Treated', 'MAE', str(mean_absolute_error(yhat_DNN,std_df_treated_y.values)))
                    
                          
            get_family_MSE(test_plate, task_channel, "DNN", "Treated", yhat_DNN,std_df_treated_y)
            get_family_MAE(test_plate, task_channel, "DNN", "Treated", yhat_DNN,std_df_treated_y)
            
            print('profile_mock:')            
            yhat_DNN = pd.DataFrame(DNN_model.predict(std_df_mock_x.values),columns=std_df_mock_y.columns)   
            print(('DNN MSE: {}'.format(mean_squared_error(yhat_DNN,std_df_mock_y.values))))  
            print(('DNN MAE: {}'.format(mean_absolute_error(yhat_DNN,std_df_mock_y.values))))  
                          
                          
            print_results(test_plate, task_channel, 'Overall', 'DNN', 'Mock', 'MSE', str(mean_squared_error(yhat_DNN,std_df_mock_y.values)))
            print_results(test_plate, task_channel, 'Overall', 'DNN', 'Mock', 'MAE', str(mean_absolute_error(yhat_DNN,std_df_mock_y.values)))
                    
                          
            get_family_MSE(test_plate, task_channel, "DNN", "Mock", yhat_DNN,std_df_mock_y)
            get_family_MAE(test_plate, task_channel, "DNN", "Mock", yhat_DNN,std_df_mock_y)
            #get_family_MSE(yhat_DNN,std_df_mock_y)
            #get_family_MAE(yhat_DNN,std_df_mock_y)
            print('**************')


# # Main

# In[35]:


main('/storage/users/AmitAdirProject/Data/sqlite/','Std')

