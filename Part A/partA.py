##################
##--- import ---##
##################

import numpy as np
import pandas as pd
import seaborn as sns
import random

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode,iplot
from plotly import tools

# show graphs and plots in browser
import plotly.io as pio
pio.renderers.default='browser'

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# for correlation
from dython.nominal import associations
from dython.nominal import identify_nominal_columns

# KNN for missing values
from sklearn.impute import KNNImputer

# for feature representation 
from sklearn.preprocessing import MinMaxScaler
import category_encoders as ce

#####################
##--- read data ---##
#####################

missing_values = ["n/a", "na", "--","nan","Nan","[None]","", "NaN"]
df = pd.read_csv('C:/Users/User/Desktop/ML_project/XY_train.csv', header = 0, na_values = missing_values)

# we set our max columns to none so we can view every column in the dataset
pd.options.display.max_columns = None 

# check if ID is unique, if 0 --> unique. 
print(df.enrollee_id.duplicated().sum())

# enrollee_id is a dataset artifact, not something useful for analysis
df.drop("enrollee_id", axis=1, inplace=True)

# information about dataset regards to missing values and datatypes
df.info()


#####################################################################################################################
' 1: Exploratory data analysis '
#####################################################################################################################

# summary table of the data
print(df.describe(include = 'all'))
df[['city_development_index', 'training_hours', 'target']].describe()

# how many employees in each discipline
print(df['major_discipline'].value_counts()/15326)

# how many employees changed their job after trainging
print(df['target'].value_counts()/15326) 


## --city_development_index-- ##
# density 
fig, ax = plt.subplots()
sns.kdeplot(df[:]["city_development_index"], shade=True, color="lightpink",ax=ax)
#sns.kdeplot(df[df["target"]==0]["training_hours"], shade=True, color="green", label="did not change", ax=ax)
ax.set_xlabel("city_development_index hours")
ax.set_ylabel("Density")
fig.suptitle("city_development_index density")

# boxplot 
boxPlot=plt.axes()
sns.boxplot(y='city_development_index', data=df, palette='pink')
boxPlot.set_title('city development index Boxplot')
plt.show()


##        --gender--       ##
sns.catplot(x="gender", kind="count", palette="ch:.25", data=df)

## --relevent_experience --##
sns.catplot(x="relevent_experience", kind="count", palette="ch:.25", data=df)

## --enrolled_university-- ##
sns.catplot(x="enrolled_university", kind="count", palette="ch:.25", data=df)

## --  education_level  -- ##
sns.catplot(x="education_level", kind="count", palette="ch:.25", order=["Primary School", "High School", "Graduate", "Masters", "Phd"], data=df)

## --  major_discipline -- ##
sns.catplot(x="major_discipline", kind="count", palette="ch:.25", data=df)

## --    experience     -- ##
sns.catplot(x="experience", kind="count", palette="ch:.25", order=["<1", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", ">20"], data=df)

## --   company_size--     ##
sns.catplot(x="company_size", kind="count", palette="ch:.25", data=df)

## --   company_type--     ##
sns.catplot(x="company_type", kind="count", palette="ch:.25", data=df)

## --   last_new_job    -- ##
sns.catplot(x="last_new_job", kind="count", palette="ch:.25", order=["never", "1", "2", "3", "4", ">4"], data=df)


##  --   training_hours -- ##
# density
fig, ax = plt.subplots()
sns.kdeplot(df[:]["training_hours"], shade=True, color="lightpink",ax=ax)
#sns.kdeplot(df[df["target"]==0]["training_hours"], shade=True, color="green", label="did not change", ax=ax)
ax.set_xlabel("training hours")
ax.set_ylabel("Density")
fig.suptitle("training hours density")

# boxplot 
boxPlot=plt.axes()
sns.boxplot(y='training_hours', data=df, palette='pink')
boxPlot.set_title('training_hours Boxplot')
plt.show()


## -- correlation between features, both continous and categorial -- ##
categorical_features = identify_nominal_columns(df)
complete_correlation = associations(df, filename= 'complete_correlation.png', figsize=(10,10))
df_complete_corr = complete_correlation['corr']
df_complete_corr.dropna(axis=1, how='all').dropna(axis=0, how='all').style.background_gradient(cmap='coolwarm', axis=None).set_precision(2)

#####################################################################################################################
' 2: Dataset creation '
#####################################################################################################################

#-------------------------------------------------------------------------------------------------------------------
' 2.1.1 pre processing - clean noise ' 
df['company_size'].replace('Oct-49','10-49',inplace=True)
df['company_size'].value_counts() # samples are corrected
print(df.groupby(['education_level', 'major_discipline']).size()) # No major for school level

#-------------------------------------------------------------------------------------------------------------------
' 2.1.2 pre processing - Redundancy in data ' 
# print(df['enrollee_id'].nunique()) # --> every sample is unique

#-------------------------------------------------------------------------------------------------------------------
' 2.1.3 pre processing - missing values ' 

# check for null values
df.isna().sum()
print(f"\nTotal missing values: {df.isna().sum().sum()}")

# 1. drop 'city' column due to maximun corr with 'city_development_index' column
#    also, drop 'gender' column due to redundant corr with 'target' column 
df = df.drop(columns = ['city', 'gender'])

# 2. with 4 or more missing values --> deleting entire sample
df = df.dropna(axis=0, thresh=8)
print(f"\nMissing Values post deleting samples and gender column: {df.isna().sum().sum()}")

# 3. fill 'No major' in major discipline for high/primary school level
df.loc[(df['major_discipline'].isna()) & (df['education_level']== "High School"), "major_discipline"] = "No Major"
df.loc[(df['major_discipline'].isna()) & (df['education_level']== "Primary School"), "major_discipline"] = "No Major"
print(df['major_discipline'].value_counts())

# 4. fill 'STEM' major to Graduates, Masters and Phd's level
df.loc[(df['major_discipline'].isna()) & (df['education_level']== "Graduate"), "major_discipline"] = "STEM"
df.loc[(df['major_discipline'].isna()) & (df['education_level']== "Masters"), "major_discipline"] = "STEM"
df.loc[(df['major_discipline'].isna()) & (df['education_level']== "Phd"), "major_discipline"] = "STEM"

# 5. fill company size and type based on most common category in the other
print(df.groupby(['company_size', 'company_type']).size())
# Pvt Ltd is most common in all company sizes
df.loc[(df['company_type'].isna()) & (df['company_size'].notna()),'company_type'] = 'Pvt Ltd'
print(df['company_type'].value_counts())

print(df.groupby(['company_type', 'company_size']).size())
# Early Stage Startup --> size = <10
# Funded Startup      --> size = 50-99
# NGO                 --> size = 100-500
# Other               --> size = 1000-4999
# Public Sector       --> size = 1000-4999
# Pvt Ltd             --> size = 50-99

df.loc[(df['company_size'].isna()) & (df['company_type']=='Early Stage Startup'),'company_size'] = '<10'
df.loc[(df['company_size'].isna()) & (df['company_type']=='Funded Startup'),'company_size'] = '50-99'
df.loc[(df['company_size'].isna()) & (df['company_type']=='NGO'),'company_size'] = '100-500'
df.loc[(df['company_size'].isna()) & (df['company_type']=='Other'),'company_size'] = '1000-4999'
df.loc[(df['company_size'].isna()) & (df['company_type']=='Public Sector'),'company_size'] = '1000-4999'
df.loc[(df['company_size'].isna()) & (df['company_type']=='Pvt Ltd'),'company_size'] = '50-99'
print(df['company_size'].value_counts())

# 6. samples with null in both 'company size' and 'company type' --> fill with 'None'
df.loc[(df['company_size'].isna()) & (df['company_type'].isna()),['company_size', 'company_type']] = 'None'

# 7. KNN for the rest of missing values - maintains the value and variability of your datasets and yet it is more precise
#    and efficient than using the average values.
#    We will need to do some pre-step before using the KNN imputer.  
print(f"\nMissing Values post completion: {df.isna().sum().sum()}")


# -------------------------------------------------------------------------------------------------------------------
' 2.2 segmentation ' 

# -------------------------------------------------------------------------------------------------------------------
' 2.3 feature extraction ' 

# 1. pop_enrollee - * no need for feature representation
df.insert (10, "pop_enrollee", 0)
df.loc[(df['major_discipline'] == 'STEM') & (df['company_type'] == 'Pvt Ltd'),'pop_enrollee'] = 1

# 2. is_working - * no need for feature representation
df.insert (11, "is_working", 1)
df.loc[(df['company_size'] == 'None') & (df['company_type'] == 'None'),'is_working'] = 0

# 3. university+relevent_exp - * needed to be nominal encoded 
df.insert (12, "university+relevent_exp", np.nan)
df.loc[(df['relevent_experience'] == 'No relevent experience') & (df['enrolled_university'] == 'no_enrollment'),'university+relevent_exp'] = 0
df.loc[(df['relevent_experience'] == 'No relevent experience') & (df['enrolled_university'] == 'Full time course'),'university+relevent_exp'] = 1
df.loc[(df['relevent_experience'] == 'Has relevent experience') & (df['enrolled_university'] == 'no_enrollment'),'university+relevent_exp'] = 2
df.loc[(df['relevent_experience'] == 'Has relevent experience') & (df['enrolled_university'] == 'Full time course'),'university+relevent_exp'] = 3
df.loc[(df['enrolled_university'] == 'Part time course'),'university+relevent_exp'] = 4

# graph
RelRatio=df.groupby(['target','relevent_experience'])['enrolled_university'].value_counts().unstack()
RelRatio.columns=RelRatio.columns.tolist()
RelRatio.reset_index(inplace=True)
fig = px.bar(RelRatio, x='relevent_experience',
             y=['Full time course','Part time course','no_enrollment'],
             color_discrete_map={'Full time course':'indigo',
                                 'Part time course':'red','no_enrollment':'limegreen'},
             facet_col="target",
             template="simple_white")
fig.update_layout(title_text="Who has more relevant experience wrt mode of education?",
        yaxis=dict(title="count"),
        annotations=[dict(text='Stay', x=0.3, y=1.0, font_size=18, showarrow=False),
                     dict(text='Change', x=0.7, y=1.0, font_size=18, showarrow=False)])
fig.update_xaxes(title ="Relevent experience")
fig.show()

# -------------------------------------------------------------------------------------------------------------------
' feature representation ' 

# 1. ordinal encoder - there is a ranked ordering between values 

# enrolled_university
df.loc[df.enrolled_university == 'no_enrollment', 'enrolled_university'] = 0
df.loc[df.enrolled_university == 'Part time course', 'enrolled_university'] = 1
df.loc[df.enrolled_university == 'Full time course', 'enrolled_university'] = 2

# education_level
df.loc[df.education_level == 'Primary School', 'education_level'] = 0
df.loc[df.education_level == 'High School', 'education_level'] = 1
df.loc[df.education_level == 'Graduate', 'education_level'] = 2
df.loc[df.education_level == 'Masters', 'education_level'] = 3
df.loc[df.education_level == 'Phd', 'education_level'] = 4

# last_new_job
df.loc[df.last_new_job == 'never', 'last_new_job'] = 0
df.loc[df.last_new_job == '1', 'last_new_job'] = 1
df.loc[df.last_new_job == '2', 'last_new_job'] = 2
df.loc[df.last_new_job == '3', 'last_new_job'] = 3
df.loc[df.last_new_job == '4', 'last_new_job'] = 4
df.loc[df.last_new_job == '>4', 'last_new_job'] = 5

# experience - we will encode the '<1' and '>21' --> convert to int --> later we will normalize
df.loc[df.experience == '<1', 'experience'] = 0
df.loc[df.experience == '>20', 'experience'] = 21
df["experience"] = df["experience"].fillna(-1)
df["experience"] = df["experience"].astype(int)
df["experience"] = df["experience"].replace(-1, np.nan)
print(df.dtypes)

# company_size
df.loc[df.company_size == '<10', 'company_size'] = 0
df.loc[df.company_size == '10-49', 'company_size'] = 1
df.loc[df.company_size == '50-99', 'company_size'] = 2
df.loc[df.company_size == '100-500', 'company_size'] = 3
df.loc[df.company_size == '500-999', 'company_size'] = 4
df.loc[df.company_size == '1000-4999', 'company_size'] = 5
df.loc[df.company_size == '5000-9999', 'company_size'] = 6
df.loc[df.company_size == '10000+', 'company_size'] = 7
df.loc[df.company_size == 'None', 'company_size'] = 8


# 2. nominal encoder 

# relevent_experience - binary feature
df.loc[df.relevent_experience == 'No relevent experience', 'relevent_experience'] = 0
df.loc[df.relevent_experience == 'Has relevent experience', 'relevent_experience'] = 1

# major_discipline 
encoder = ce.BaseNEncoder(cols=['major_discipline'],return_df=True,base=5)
df = encoder.fit_transform(df)

# company_type - we will trasform to a binary feature - 1->Pvt Ltd, 0->else
encoder = ce.BaseNEncoder(cols=['company_type'],return_df=True,base=5)
df = encoder.fit_transform(df)

# university+relevent_exp
encoder = ce.BaseNEncoder(cols=['university+relevent_exp'],return_df=True,base=5)
df = encoder.fit_transform(df)

# 3. normalization 

# Another critical point here is that the KNN Imptuer is a distance-based imputation method and it requires us to normalize our data
# Otherwise, the different scales of our data will lead the KNN Imputer to generate biased replacements for the missing values
# For simplicity, we will use Scikit-Learn’s MinMaxScaler which will scale our variables to have values between 0 and 1
scaler = MinMaxScaler()
df[['training_hours']] = scaler.fit_transform(df[['training_hours']])
df[['experience']] = scaler.fit_transform(df[['experience']])
df.head()


# -------------------------------------------------------------------------------------------------------------------
' pre processing - missing values - continue with KNN ' 

# Now our dataset is encoded and normalized, so we can use KNN imputation
# we are setting the parameter ‘n_neighbors’ as 5 i.e the missing values will be replaced by the mean value of 5 nearest neighbors measured by Euclidean distance\
imputer = KNNImputer(n_neighbors = 1)
df = pd.DataFrame(imputer.fit_transform(df),columns = df.columns)

# the results
df.isna().any()


# -------------------------------------------------------------------------------------------------------------------
' feature selection '

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from matplotlib import pyplot

def select_features(X_train, y_train):
	fs = SelectKBest(score_func = mutual_info_classif, k='all')
	fs.fit(X_train, y_train)
	return fs

X_train = df.drop('target', 1).values
y_train = df['target'].values

fs = select_features(X_train, y_train)

# what are scores for the features
for i in range(len(fs.scores_)):
	print('Feature %d: %f' % (i, fs.scores_[i]))

# for sorting only 
for i in range(len(fs.scores_)):
	print(fs.scores_[i])
   
# plot the scores
pyplot.bar([i for i in range(len(fs.scores_))], fs.scores_)
pyplot.show()

df = df.drop(columns = ['enrolled_university', 'education_level', 'major_discipline_0', 'major_discipline_1', 'company_type_0', 'last_new_job', 'training_hours' ])

# -------------------------------------------------------------------------------------------------------------------
' dimensionality Reduction '

from sklearn.decomposition import PCA
from sklearn import preprocessing

# PCA
dfWithoutViews = df[['city_development_index','relevent_experience','company_size', 'company_type_1', 'is_working', 'university+relevent_exp_0', 'university+relevent_exp_1', 'experience', 'pop_enrollee' ]]
dataScaled = preprocessing.scale(dfWithoutViews) # scaling the data to be with mean=0 and sd=1
pca = PCA(0.8) 
pca.fit(dataScaled)
pca_data=pca.transform(dataScaled)
var=np.round(pca.explained_variance_ratio_*100 , decimals=1)
labels = ['PC'+str(x) for x in range(1, len(var)+1)]
plt.bar(x=range(1,len(var)+1) ,height=var , tick_label=labels)
plt.ylabel('% of Explained Variance')
plt.xlabel('PCi')
plt.title('PCA Plot')
plt.show()
PCi=pca.transform(dataScaled)
print(PCi)
print(var)