#!/usr/bin/env python
# coding: utf-8

# # Topic

# The topic choose here is the wine quality. We are using two datasets with red and white wine samples.
# The inputs include objective tests (for example, PH values) and the output is based on sensory data (median of at least 3 evaluations made by wine experts). 
# 
# Each expert graded the wine quality between 0 (very bad) and 10 (very excellent).

# # Objectif

# In this project we will try to identify, via the linear regression method, the quality of the wine (our target) among the features of the dataset : 

# # Import librairies

# In[450]:


import pandas as pd
from pandas_visual_analysis import VisualAnalysis
import seaborn as sns
from visualizer import Visualizer

import matplotlib.pyplot as plt
import seaborn as sns


# # Manipulating

# ## Read files and concatenate

# In[451]:


#dataset red wine and white wine
url1='/Users/tatiana/WineQuality/data/winequality-red.csv'
url2='/Users/tatiana/WineQuality/data/winequality-white.csv'


# In[452]:


#read the csv files
df1=pd.read_csv(url1)
df2=pd.read_csv(url2)


# In[453]:


#look at the dataframe
df1


# In[454]:


df2


# In[455]:


df1.columns


# In[456]:


df2.columns


# All the data are located in the column :
#     split all after ";"
# In the column names :
#     1.replace le space by a "_"
#     2.delete "" for some of them

# ## Concatenate files

# In[457]:


df_wines=df1.append(df2)


# In[458]:


df_wines


# In[459]:


df_rows = df1.shape[0]+df2.shape[0]
df_wines_rows = df_wines.shape[0]

if df_rows == df_wines_rows :
    print('Good! the number of rows expected and the number of the new dataframe are the same')
else:
    print('Grrr, pb of number of rows!')
print(f'After having concontenate both dataframes, number of rows is {df_wines_rows} and\na sum of rows of both dataframe is {df_rows}')


# ## Split

# Now, split (;) all the dataframe

# In[460]:


#split the dataframe into columns.
df_wine = df_wines.iloc[:,0].str.split(';', expand=True)


# In[461]:


df_wine


# Columns has been created : OK. We need names for the columns. We will use those from our original dataframe

# In[462]:


#give the names of the columns of the df_wine thanks to the original dataframe
df_wine.columns = df1.columns[0].split(';')


# In[463]:


df_wine.head()


# ## Add a column in the dataframe - color of the wine : red or white

# In[464]:


#add new column in the 1st position : color with 2 unique values, red or white
df_wine.insert(0, "color", "Red") 


# In[465]:


df_wine.columns


# The column color has been added in the 1rst place and fill with red value, with success.
# Now we will replace red by white in the column color for the white win of the dataframe

# In[466]:


last_row_red = df1.shape[0]#last row in the dataframe of red wine

df_wine.loc[last_row_red:,'color'] = "white"#replace white by red in the dataframe wine
df_wine


# # Cleanning

# ## Columns

# In[467]:


#replace spaces in the names'columns by _
df_wine.columns = df_wine.columns.str.replace(' ', '_')


# In[468]:


df_wine


# Spaces in the columns have been replaced by _ >> OK

# In[469]:


#delete "" in the names'columns 
df_wine.columns = df_wine.columns.str.replace('"', '')


# In[470]:


df_wine


# " in the columns have been deleted >> OK

# ## Missing values

# In[471]:


na=df_wine.isna().sum()


# In[472]:


na


# The dataframe doen't have missing values >> Ok, let's continue

# # Dummies : categorical column color

# In[473]:


#Transform the color column into dummies. Drop the first column of dummies, use drop_first=True.
df_dum=pd.get_dummies(data=df_wine, columns=['color'], drop_first=True)


# In[474]:


df_dum


# Now the dummy dataframe, df_dum, has 0 for Red wine and 1 for white wine.

# The color column has been renamed color_white. Let's change the name by dummy_color.

# # Rename the dummy column

# In[475]:


df_dum.rename(columns={"color_white": "dummy_color"}, inplace=True)


# In[476]:


df_dum.columns


# the name has been correctly changed.

# # Dataframe info

# In[477]:


df_dum.info()


# The columns are "object", we need numerical for the statistical Analysis with Linear Regression.
# 
# Our original dataframe had 12 columns. Now, after manipulating, the dataframe df_dum has 13 columns (12:original  2:dummies - 1:color_red). The new number of columns is correct.

# In[478]:


df_dum.describe()


# The describe of the dataframe for object and for uint8 give different information.
# For our linear regression, we need only numerical datas.
# 
# By looking at the average of each characteristic of the wine and comparing it with the other values in the "describe", we do not observe any apparent inconsistency.
# 

# ## Convert all the object data into numerical data

# In[479]:


col=df_dum.columns.tolist()


# In[480]:


col


# In[481]:


def to_numeric(df=df_dum):
    #list of columns in the dataframe
    col=df.columns.tolist()
    #loop in each column and convert all object into numerical
    for i in col:
        df[i]=df[i].astype(float)
        pass


# In[482]:


to_numeric()


# In[483]:


df_dum.info()


# Good! Dtype in the dataframe df_wine is now float64

# # Explorating Data Analysis by visualization 

# ## Correlation

# In[484]:


#display your viz after running the code (matplot-jupyter)
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

import warnings
warnings.simplefilter('ignore')

get_ipython().run_line_magic('config', "InlineBackend.figure_format='svg'")

#Test the multicollinearity
def heatmap(df=df_dum,title="Wine"):
    #new dataframe HeatMap:
    df_HeatMap = pd.DataFrame(df,columns=df.columns)

    #plot HeatMap with title and map param:
    plt.title(f'Heatmap - {title} correlation', fontsize =14,loc='center')
    heatmap = sns.heatmap(df.corr(), cmap="PiYG", robust=True,annot=True,annot_kws={'size':5},fmt=".1f",vmin=-1, vmax=1)
    square=True

    return heatmap


# In[485]:


heatmap(df=df_dum,title="Wine")


# In this heatmap, we don't see high correlation between the quality and other datas.
# 
# In descending order, the 3 strongest correlations with our target, the 'quality' of wine are :
#     - alcohol          (0.4)
#     - volatile acidity (-0.3)
#     - density          (-0.3)
#     
# In descending order, we notice a strong correlation between the features themselves : 
#     - red and white wine (-1)
#     - alcohol and density (-0.7)
#     - density and residual_sugar (-0.6)
#     - density and fixed_acid (-0.5)
#     - total_sulfure_dioxide and residual_sugar (-0.5)
# 
# 

# ## Visual Analysis

# In[486]:


#pip install visualizer


# In[487]:


import pandas as pd
from pandas_visual_analysis import VisualAnalysis
import seaborn as sns
VisualAnalysis(df_dum)


# Selecting a graph with x = feature and y=Quality, we never see linear regression. We see only 
# some parallel lines.
# We also note that the distribution of the quality does not follow a normal distribution.

# It can be interesting to test if there is a difference of quality between red and white wine

# # Quality of red wine - Quality of white wine

# In[488]:


df_red=df1.iloc[:,0].str.split(';', expand=True)


# In[489]:


df_red.columns = df1.columns[0].split(';')


# In[490]:


df_red.columns = df_red.columns.str.replace(' ', '_')


# In[491]:


df_red.columns = df_red.columns.str.replace('"', '')


# In[492]:


to_numeric(df=df_red)


# In[493]:


df_red.info()


# In[494]:


df_white=df2.iloc[:,0].str.split(';', expand=True)


# In[495]:


df_white.columns = df2.columns[0].split(';')


# In[496]:


df_white.columns = df_white.columns.str.replace(' ', '_')


# In[497]:


df_white.columns = df_white.columns.str.replace('"', '')


# In[498]:


to_numeric(df=df_white)


# In[499]:


df_white.info()


# ## correlation red wine

# In[500]:


heatmap(df=df_red,title="Red Wine")


# In[501]:


heatmap(df=df_white,title="White Wine")


# We notice some differencies in the heatmap correlation between red wine and white wine.
# Not too much between quality and other features but between the features themselves.

# ## Visual Analyses red wine

# In[355]:


import pandas as pd
from pandas_visual_analysis import VisualAnalysis
import seaborn as sns
VisualAnalysis(df_red)


# ## Visual Analyses white wine

# In[281]:


import pandas as pd
from pandas_visual_analysis import VisualAnalysis
import seaborn as sns
VisualAnalysis(df_white)


# # Modelling

# Import librairies

# In[502]:


from statsmodels.formula.api import ols
from statsmodels.api import OLS


# In[503]:


df_dum.head()


# We test the model addind feature one by one.

# ## Test with feature alcohol

# In[504]:


ols('quality ~ alcohol', data=df_dum).fit().summary()


# Even if R-squared is low : 0.20 (approx.), it is closed to Adj. R-squared value. It is a good point.
# The P(t) is approximatively null so, we can consider that we have a linear relationship between the quality of wine (target) and the alcohol (feature).

# ## Test with feature alcohol and sulphates

# In[505]:


ols('quality ~ alcohol+sulphates', data=df_dum).fit().summary()


# R-squared is always low : 0.20 (approx.) but it is closed to Adj. R-squared value. It is a good point. 
# F-statistic: 806.7 is high and Prob (F-statistic) is close to 0(P(F-statistic)<0.05). There is a good amount of linear relationship between my target and the features.
# 
# The P(t) is approximatively null so, we can consider that we have a linear relationship between the quality of wine (target) and sulphates (feature).

# ## Test with feature alcohol, sulphates and pH

# In[506]:


ols('quality ~ alcohol+sulphates+pH', data=df_dum).fit().summary()


# R-squared is always low : 0.20 (approx.) but it is closed to Adj. R-squared value. It is a good point. 
# F-statistic: 544.0 is high and Prob (F-statistic) is close to 0(P(F-statistic)<0.05). There is a good amount of linear relationship between my tardet and the features.
# 
# The P(t) is approximatively null so, we can consider that we have a linear relationship between the quality of wine (target) and pH (feature).

# ## Test with feature alcohol, sulphates, pH, density

# In[507]:


ols('quality ~ alcohol+volatile_acidity+density', data=df_reg).fit().summary()


# R-squared is always low : 0.27 (approx.) and it is closed to Adj. R-squared value. It is a good point. 
# F-statistic: 789.6 is high and Prob (F-statistic) is close to 0(P(F-statistic)<0.05). There is a good amount of linear relationship between my target and the features.
# 
# The P(t) is approximatively null so, we can consider that we have a linear relationship between the quality of wine (target) and density (feature).
# 
# But now, a note has appeared indicating that there are strong multicollinearity or other numerical problems. This note has appeared by adding the feature density so we have to drop it.

# ## Test with feature alcohol, sulphates, pH and free_sulfur_dioxide

# In[508]:


ols('quality ~ alcohol+sulphates+pH+free_sulfur_dioxide', data=df_dum).fit().summary()


# R-squared is always low : 0.22 (approx.) and it is closed to Adj. R-squared value. It is a good point. F-statistic: 462.5 is high and Prob (F-statistic) is close to 0(P(F-statistic)<0.05). There is a good amount of linear relationship between my target and the features.
# 
# The P(t) is approximatively null so, we can consider that we have a linear relationship between the quality of wine (target) and free_sulfur_dioxide (feature).
# 
# P(t) for ph is now not null but is under 0.05 so it's ok.

# ## Test with feature alcohol, sulphates, pH, free_sulfur_dioxide and chlorides

# In[509]:


ols('quality ~ alcohol+sulphates+pH+free_sulfur_dioxide+chlorides', data=df_dum).fit().summary()


# R-squared is always low : 0.23 (approx.) and it is closed to Adj. R-squared value. It is a good point. F-statistic: 387.2 is high and Prob (F-statistic) is close to 0(P(F-statistic)<0.05). There is a good amount of linear relationship between my target and the features.
# 
# The P(t) is approximatively null so, we can consider that we have a linear relationship between the quality of wine (target) and free_sulfur_dioxide (feature).
# 
# P(t) for ph is now not null but is under 0.05 so it's ok.
# 
# But now, a note has appeared indicating that there are strong multicollinearity or other numerical problems. This note has appeared by adding the feature density so we have to drop it.

# ## Test with feature alcohol, sulphates, pH, free_sulfur_dioxide and residual_sugar

# In[510]:


ols('quality ~ alcohol+sulphates+pH+free_sulfur_dioxide+residual_sugar', data=df_dum).fit().summary()


# R-squared is always low : 0.23 (approx.) and it is closed to Adj. R-squared value. It is a good point. F-statistic: 462.5 is high and Prob (F-statistic) is close to 0(P(F-statistic)<0.05). There is a good amount of linear relationship between my target and the features.
# 
# The P(t) is approximatively null so, we can consider that we have a linear relationship between the quality of wine (target) and residual_sugar (feature).
# 
# P(t) for ph is now not null but is over 0.05 so we have to drop it in order to continue with residual_sugar

# ## Test with feature alcohol, sulphates, free_sulfur_dioxide, residual_sugar and citric_acid

# In[511]:


ols('quality ~ alcohol+sulphates+free_sulfur_dioxide+residual_sugar+citric_acid', data=df_dum).fit().summary()


# R-squared is always low : 0.23 (approx.) and it is closed to Adj. R-squared value. It is a good point. F-statistic: 395.5 is high and Prob (F-statistic) is close to 0(P(F-statistic)<0.05). There is a good amount of linear relationship between my target and the features.
# 
# The P(t) is approximatively null so, we can consider that we have a linear relationship between the quality of wine (target) and citric_acid (feature).

# ## Test with feature alcohol, sulphates, free_sulfur_dioxide, residual_sugar, citric_acid and volatile_acidity

# In[512]:


ols('quality ~ alcohol+sulphates+free_sulfur_dioxide+residual_sugar+citric_acid+volatile_acidity', data=df_dum).fit().summary()


# R-squared is always low : 0.28 (approx.) and it is closed to Adj. R-squared value. It is a good point. F-statistic: 421.1 is high and Prob (F-statistic) is close to 0(P(F-statistic)<0.05). There is a good amount of linear relationship between my target and the features.
# 
# The P(t) is approximatively null so, we can consider that we have a linear relationship between the quality of wine (target) and volatile_acidity (feature).
# 
# P(t) for free_sulfur_dioxide and citric_acid are not null anymore but are under 0.05. We can continue.

# ## Test with feature alcohol, sulphates, free_sulfur_dioxide, residual_sugar, citric_acid, volatile_acidity and fixed_acidity

# In[513]:


ols('quality ~ alcohol+sulphates+free_sulfur_dioxide+residual_sugar+citric_acid+volatile_acidity+fixed_acidity', data=df_dum).fit().summary()


# R-squared is always low : 0.28 (approx.) and it is closed to Adj. R-squared value. It is a good point. F-statistic: 361.9 is high and Prob (F-statistic) is close to 0(P(F-statistic)<0.05). There is a good amount of linear relationship between my target and the features.
# 
# The P(t) is approximatively null so, we can consider that we have a linear relationship between the quality of wine (target) and fixed_acidity (feature).
# 
# P(t) for fixed_acidity is not null anymore but is under 0.05. 

# ## Test with feature alcohol, sulphates, free_sulfur_dioxide, residual_sugar, citric_acid, volatile_acidity,fixed_acidity, dummy_color

# In[515]:


ols('quality ~ alcohol+sulphates+free_sulfur_dioxide+residual_sugar+citric_acid+volatile_acidity+fixed_acidity+dummy_color', data=df_dum).fit().summary()


# R-squared is always low : 0.29 (approx.) and it is closed to Adj. R-squared value. It is a good point. F-statistic: 330.0 is high and Prob (F-statistic) is close to 0(P(F-statistic)<0.05). There is a good amount of linear relationship between my target and the features.
# 
# The P(t) is approximatively null so, we can consider that we have a linear relationship between the quality of wine (target) and fixed_acidity (feature).
# 
# P(t) for fixed_acidity is not null anymore but is over 0.05.

# # Models

# 1. ols('quality ~ alcohol+sulphates+free_sulfur_dioxide+residual_sugar+citric_acid+volatile_acidity+fixed_acidity', data=df_dum).fit().summary()
# 
# 
# 2. ols('quality ~ alcohol+sulphates+free_sulfur_dioxide+residual_sugar+citric_acid+volatile_acidity+dummy_color', data=df_dum).fit().summary()
# 
# 
# 3. ols('quality ~ alcohol+sulphates+pH+free_sulfur_dioxide+chlorides', data=df_dum).fit().summary()

# # Red Wine

# ## test with alcohol+sulphates+free_sulfur_dioxide+residual_sugar+citric_acid+volatile_acidity+fixed_acidity

# In[368]:


ols('quality ~ alcohol+sulphates+free_sulfur_dioxide+residual_sugar+citric_acid+volatile_acidity+fixed_acidity', data=df_red).fit().summary()


# R-squared is always low : 0.34 (approx.) and it is closed to Adj. R-squared value. It is a good point. F-statistic: 119.4 is high and Prob (F-statistic) is close to 0(P(F-statistic)<0.05). There is a good amount of linear relationship between my target and the features.
# 
# The P(t) are approximatively null so for  most of the features butt not for free_sulfur_dioxide and residual_sugar.

# ## Test with alcohol+sulphates+pH+free_sulfur_dioxide+chlorides

# In[516]:


ols('quality ~ alcohol+sulphates+pH+free_sulfur_dioxide+chlorides', data=df_red).fit().summary()


# R-squared is always low : 0.30 (approx.) and it is closed to Adj. R-squared value. It is a good point. F-statistic: 138.4 is high and Prob (F-statistic) is close to 0(P(F-statistic)<0.05). There is a good amount of linear relationship between my target and the features.
# 
# The P(t) are approximatively null so for most of the features butt not for free_sulfur_dioxide.

# ## Test with alcohol+sulphates+pH+chlorides

# In[ ]:


alcohol+sulphates+pH+chlorides


# In[518]:


ols('quality ~ alcohol+sulphates+pH+chlorides', data=df_red).fit().summary()


# ## Test with alcohol+sulphates+citric_acid+volatile_acidity+fixed_acidity

# In[519]:


ols('quality ~ alcohol+sulphates+citric_acid+volatile_acidity+fixed_acidity', data=df_red).fit().summary()


# # White Wine - test with alcohol+sulphates+free_sulfur_dioxide+residual_sugar+citric_acid+volatile_acidity+fixed_acidity

# In[370]:


ols('quality ~ alcohol+sulphates+free_sulfur_dioxide+residual_sugar+volatile_acidity+fixed_acidity', data=df_white).fit().summary()


# The model is ok with white wine.
# 
# 
# There are 3 times more lines for white wines than for red wines. Our model has been tested on the entire dataframe (white and red wines). The white wine is preponderant, it is normal that the model works very well with the dataframe which contains only the white wines.

# # P Haking

# In[374]:


def foo(col, X, y=y):
    if col:
        X1=X.drop(col, axis=1).copy()
    else:
        X1=X
    display(OLS(y,add_constant(X1)).fit().summary())
    return X1


# In[372]:


#we drop chlorides column because of the pvalue
foo('chlorides', X, y=y)


# In[373]:


foo('citric_acid', X, y=y)


# # Assumptions

# ## Class for testing the Assumptions

# In[389]:


class Assumption_Tester_OLS:
    """
    X - Pandas DataFrame with numerical values. Independent Variable
    y - Series with numerical values. Dependent Variable
    
    Tests a linear regression on the model to see if assumptions are being met

    """
    
    from sklearn.linear_model import LinearRegression
    
    def __init__(self, X,y):
        from numpy import ndarray
        from pandas import concat
        from pandas.core.frame import DataFrame
        from pandas.core.series import Series

        if type(X) == ndarray:
            self.features = ['X'+str(feature+1) for feature in range(X.shape[1])]
        elif type(X) == DataFrame:
            self.features=X.columns.to_list()
        else:
            print('Expected numpy array or pandas dataframe as X')
            return
        if type(y) == ndarray:
            self.output = 'y'
        elif type(y) == DataFrame:
            self.output=y.columns[0]
        elif type(y) == Series:
            self.output=y.name
        else:
            print('Expected numpy array or pandas dataframe as X')
            return

        self.X = X.values if type(X)==DataFrame else X
        self.y=y.iloc[:,0].values if type(y)==DataFrame else y.values if type(y)==Series else y
        
        self.model='not built yet'
        self.r2=0
        self.results={'Satisfied':[],'Potentially':[],'Violated':[]}
    
    def fit_model(self):
        from sklearn.linear_model import LinearRegression
        
        print('Fitting linear regression')        
        
        #Multi-threading when needed
        if self.X.shape[0] > 100000:
            self.model = LinearRegression(n_jobs=-1)
        else:
            self.model = LinearRegression()
        self.model.fit(self.X, self.y)
        
        self.predictions = self.model.predict(self.X)
        self.resid = self.y - self.predictions
        
        
    def build_model(self):
        self.fit_model()
        
        # Returning linear regression R^2 and coefficients before performing diagnostics
        self.r2 = self.model.score(self.X, self.y)
        print()
        print(('R^2:', self.r2, '\n'))
        print('Coefficients')
        print('-------------------------------------')
        print(('Intercept:', self.model.intercept_))
        for idx,feature in enumerate(self.model.coef_):
            print(f'{self.features[idx]}: {round(feature,2)}')

    def linearity(self):
        """
        Linearity: Assumes there is a linear relationship between the predictors and
                   the response variable. If not, either a polynomial term or another
                   algorithm should be used.
        """
        from pandas import concat
        from numpy import arange
        from pandas.core.frame import DataFrame
        from pandas.core.series import Series        
        import seaborn as sns
        sns.set()
        import matplotlib.pyplot as plt
        
        if type(self.model)==str:
            self.fit_model()
        
        print('\n=======================================================================================')
        print('Assumption 1: Linear Relationship between the Target and the Features')
        print('Checking with a scatter plot of actual vs. predicted. Predictions should follow the diagonal line.')
        
        # Plotting the actual vs predicted values
        sns.regplot(self.y,self.predictions, fit_reg=False)
        
        # Plotting the diagonal line
        line_coords = arange(min(self.y.min(),self.predictions.min()), max(self.y.max(),self.predictions.max()))
        plt.plot(line_coords, line_coords,  # X and y points
                 color='darkorange', linestyle='--')
        plt.title('Actual vs. Predicted')
        plt.show()
        print('If non-linearity is apparent, consider adding a polynomial term \n\t\tor using box-cox transformation to make X or y follow normal distribution')
        
        print('\n\n\nBuilding a correlation table')
        print('\n=======================================================================================')
        df=concat([DataFrame(self.X),Series(self.y)],axis=1)
        df.columns=self.features+[self.output]
        df_corr=df[df.nunique()[df.nunique()>2].index].corr()[self.output].drop(self.output)
        
        print(f'\nParameters that are most likely VIOLATE linearity assumption and their correlation with {self.output}')
        display(df_corr[abs(df_corr)<0.25])

        print(f'\nParameters that are most likely FOLLOW linearity assumption and their correlation with {self.output}')
        display(df_corr[abs(df_corr)>=0.25])
        
        
        if df_corr[abs(df_corr)<0.25].shape[0]==0:
            self.results['Satisfied'].append('Linearity')
        elif df_corr[abs(df_corr)>=0.25].shape[0]==0:
            self.results['Violated'].append('Linearity')
        else:
            self.results['Potentially'].append('Linearity')
        
    def multicollinearity(self):
        """
        Multicollinearity: Assumes that predictors are not correlated with each other. If there is
                           correlation among the predictors, then either remove prepdictors with high
                           Variance Inflation Factor (VIF) values or perform dimensionality reduction
                           This assumption being violated causes issues with interpretability of the 
                           coefficients and the standard errors of the coefficients.
        """
        from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
        import matplotlib.pyplot as plt
        import seaborn as sns
        from pandas.core.frame import DataFrame
        sns.set()
        
        if type(self.model)==str:
            self.fit_model()
            
        print('\n=======================================================================================')
        print('Assumption 2: Little to no multicollinearity among predictors')
        # Plotting the heatmap
        plt.figure(figsize = (10,8))
        sns.heatmap(DataFrame(self.X, columns=self.features).corr(), annot=len(self.features)<10, center=0, cmap=sns.diverging_palette(220, 20, as_cmap=True))
        plt.title('Correlation of Variables')
        plt.show()
        print('Variance Inflation Factors (VIF)')
        print('> 10: An indication that multicollinearity may be present')
        print('> 100: Certain multicollinearity among the variables')
        print('-------------------------------------')
        # Gathering the VIF for each variable
        vifs = {i:VIF(self.X, idx) for idx,i in enumerate(self.features)}
        vifs = dict(sorted(list(vifs.items()), key=lambda x: x[1], reverse=True))
        for key, vif in list(vifs.items()):
            print(f'{key}: {vif}')
        # Gathering and printing total cases of possible or definite multicollinearity
        possible_multicollinearity = sum([1 for vif in list(vifs.values()) if vif > 10])
        definite_multicollinearity = sum([1 for vif in list(vifs.values()) if vif > 100])
        print()
        print(f'{possible_multicollinearity} cases of possible multicollinearity')
        print(f'{definite_multicollinearity} cases of definite multicollinearity')
        print()
        if definite_multicollinearity == 0:
            if possible_multicollinearity == 0:
                print('Assumption satisfied')
                self.results['Satisfied'].append('Multicollinearity')
            else:
                print('Assumption possibly satisfied')
                print()
                print('Coefficient interpretability may be problematic')
                print('Consider removing variables with a high Variance Inflation Factor (VIF)')
                self.results['Potentially'].append('Multicollinearity')

        else:
            print('Assumption not satisfied')
            print()
            print('Coefficient interpretability will be problematic')
            print('Consider removing variables with a high Variance Inflation Factor (VIF)')
            self.results['Violated'].append('Multicollinearity')
            

    
    def autocorrelation(self):
        """
        Autocorrelation: Assumes that there is no autocorrelation in the residuals. If there is
                         autocorrelation, then there is a pattern that is not explained due to
                         the current value being dependent on the previous value.
                         This may be resolved by adding a lag variable of either the dependent
                         variable or some of the predictors.
        """
        from statsmodels.stats.stattools import durbin_watson        
        
        if type(self.model)==str:
            self.fit_model()
        print('\n=======================================================================================')
        print('Assumption 3: No Autocorrelation')
        print('\nPerforming Durbin-Watson Test')
        print('Values of 1.5 < d < 2.5 generally show that there is no autocorrelation in the data')
        print('0 to 2< is positive autocorrelation')
        print('>2 to 4 is negative autocorrelation')
        print('-------------------------------------')
        durbinWatson = durbin_watson(self.resid)
        print(('Durbin-Watson:', durbinWatson))
        if durbinWatson < 1.5:
            print(('Signs of positive autocorrelation', '\n'))
            print(('Assumption not satisfied', '\n'))
            self.results['Violated'].append('Autocorrelation')
        elif durbinWatson > 2.5:
            print(('Signs of negative autocorrelation', '\n'))
            print(('Assumption not satisfied', '\n'))
            self.results['Violated'].append('Autocorrelation')
        else:
            print(('Little to no autocorrelation', '\n'))
            print('Assumption satisfied')
            self.results['Satisfied'].append('Autocorrelation')
            

    def homoskedasticity(self,p_value_thresh=0.05):
        """
        Homoskedasticity: Assumes that the errors exhibit constant variance
        """
        
        from statsmodels.stats.diagnostic import het_breuschpagan
        
        import matplotlib.pyplot as plt
        import seaborn
        from numpy import repeat
        seaborn.set()
        
        if type(self.model)==str:
            self.fit_model()
            
        print('\n=======================================================================================')
        print('Assumption 4: Homoskedasticity of Error Terms')
        print('Residuals should have relative constant variance')
        # Plotting the residuals
        plt.subplots(figsize=(12, 6))
        ax = plt.subplot(111)  # To remove spines
        plt.scatter(x=list(range(self.X.shape[0])), y=self.resid, alpha=0.5)
        plt.plot(repeat(0, self.X.shape[0]), color='darkorange', linestyle='--')
        ax.spines['right'].set_visible(False)  # Removing the right spine
        ax.spines['top'].set_visible(False)  # Removing the top spine
        plt.title('Residuals')
        plt.show() 
        print('If heteroskedasticity is apparent, confidence intervals and predictions will be affected')        
        print('\nConsider removing outliers and preprocessing features - nonlinear transformation can help')
        
        lnames=['Lagrange Multiplier', 'pvalue for LM','F stats','pvalue for Fstats']
        display({lnames[idx]:het_breuschpagan(self.resid,self.X)[idx] for idx in range(4)})
        if het_breuschpagan(self.resid,self.X)[3] < p_value_thresh:
            print(('Signs of positive autocorrelation', '\n'))
            print(('Assumption potentially not satisfied', '\n'))
            self.results['Potentially'].append('Autocorrelation')
        else:
            print(('Signs of negative autocorrelation', '\n'))
            print(('Assumption satisfied', '\n'))
            self.results['Satisfied'].append('Autocorrelation')

       
        
    def normality_resid(self,p_value_thresh=0.05):
        """
        Normality: Assumes that the error terms are normally distributed. If they are not,
        nonlinear transformations of variables may solve this.
        This assumption being violated primarily causes issues with the confidence intervals
        """
        from statsmodels.stats.diagnostic import normal_ad
        from scipy.stats import probplot
        import pylab
        import matplotlib.pyplot as plt
        import seaborn as sns
        from numpy import quantile,logical_or
        sns.set()

        if type(self.model)==str:
            self.fit_model()
            
        print('\n=======================================================================================')
        print('Assumption 5: The error terms are kinda normally distributed')
        print()
        print('Using the Anderson-Darling test for normal distribution')
        # Performing the test on the residuals
        p_value = normal_ad(self.resid)[1]
        print(('p-value from the test - below 0.05 generally means non-normal:', p_value))
        # Reporting the normality of the residuals
        if p_value < p_value_thresh:
            print('Residuals are not normally distributed')
        else:
            print('Residuals are normally distributed')
        # Plotting the residuals distribution
        plt.subplots(figsize=(12, 6))
        plt.title('Distribution of Residuals')
        sns.distplot(self.resid)
        plt.show()
        print()
        if p_value > p_value_thresh:
            print('Assumption satisfied')
            self.results['Satisfied'].append('Normality')
        else:
            print('Assumption not satisfied')
            self.results['Violated'].append('Normality')
            print()
            print('Confidence intervals will likely be affected')
            print('Try performing nonlinear transformations on variables')
    
    
        print('Building a probability plot')
        quantiles=probplot(self.resid, dist='norm', plot=pylab);
        plt.show()
        qqq=(quantiles[0][1]-quantiles[0][1].mean())/quantiles[0][1].std()-quantiles[0][0]
        q75=quantile(qqq,0.75)
        q25=quantile(qqq,0.25)

        outliers_share=(logical_or(qqq>q75+(q75-q25)*1.7, qqq<q25-(q75-q25)*1.7).sum()/qqq.shape[0]).round(3)
        if outliers_share<0.005:
            print('Assumption can be considered as satisfied.')
            self.results['Satisfied'].append('Sub-Normality')
        elif outliers_share<0.05:
            self.results['Potentially'].append('Sub-Normality')
            print(f'\nIn your dataset you quite fat tails. You have {outliers_share} potential outliers ({logical_or(qqq>q75+(q75-q25)*1.7, qqq<q25-(q75-q25)*1.7).sum()} rows)')
        else:
            print(f'\nIn fact outliers are super significant. Probably it is better to split your dataset into 2 different ones.')
            self.results['Violated'].append('Sub-Normality')


    def run_all(self):
        self.build_model()
        self.linearity()
        self.multicollinearity()
        self.autocorrelation()
        self.homoskedasticity()
        self.normality_resid()
        display(self.results)


# # Assumptions

# ## model with alcohol+sulphates+free_sulfur_dioxide+residual_sugar+citric_acid+volatile_acidity+fixed_acidity', data

# In[414]:


from Assumptions import Assumption_Tester_OLS as ast


# In[415]:


y=df_dum.quality# our target
X=df_dum.drop(['chlorides','total_sulfur_dioxide','density','pH','quality','dummy_color'], axis=1)# our features


# In[416]:


y.iloc[0]


# In[417]:


ast(X,y).run_all()


# Assumption 1: Linear Relationship between the Target and the Features: KO
# Parameters that are most likely FOLLOW linearity assumption and their correlation with quality
# volatile_acidity   -0.265699
# alcohol             0.444319
# 
# 
# Assumption 2: multicollinearity :OK
# 3 cases of possible multicollinearity
# 
# Assumption 3: No Autocorrelation : ok
# 
# Homoskedasticity of Error Terms : OK
# 
# Assumption 5: The error terms are kinda normally distributed: KO

# ## model with volatile_acidity and alcohol only

# In[418]:


from Assumptions import Assumption_Tester_OLS as ast


# In[423]:


y=df_dum.quality# our target
X=df_dum.drop(['chlorides','total_sulfur_dioxide','density','pH','quality','fixed_acidity','citric_acid','residual_sugar','free_sulfur_dioxide','sulphates','dummy_color'], axis=1)# our features


# In[424]:


y.iloc[0]


# In[425]:


type(y)


# In[426]:


X


# In[427]:


ast(X,y).run_all()


# In[294]:


y.unique()


# ## model with 

# In[520]:


y=df_dum.quality# our target
X=df_dum.drop(['chlorides','total_sulfur_dioxide','density','pH','quality','residual_sugar','free_sulfur_dioxide','dummy_color'], axis=1)# our features         


# In[521]:


X 
# Test with alcohol+sulphates+citric_acid+volatile_acidity+fixed_acidity


# In[522]:


ast(X,y).run_all()


# ## polynomial function apply to X

# In[286]:


y=df_reg.quality# our target
X=df_reg.drop(['density', 'pH','alcohol','quality','fixed_acidity'], axis=1)# our features


# In[291]:


# Fitting Polynomial Regression to the dataset 
from sklearn.preprocessing import PolynomialFeatures 
  
poly = PolynomialFeatures(degree = 4) 
X_poly = poly.fit_transform(X)


# # RandomForestRegressor

# In[442]:


from sklearn.ensemble import RandomForestRegressor


# In[443]:


y=df_dum.quality# our target
X=df_dum.drop(['chlorides','total_sulfur_dioxide','density','pH','quality','dummy_color'], axis=1)# our features


# In[444]:


lr=RandomForestRegressor()
lr.fit(X,y)


# In[445]:


lr.predict(X)


# In[446]:


plt.scatter(y,lr.predict(X))


# In[447]:


from sklearn.metrics import r2_score


# In[448]:


r2_score(y,lr.predict(X))


# In[449]:


confusion_matrix(y,lr.predict(X))


# In[ ]:




