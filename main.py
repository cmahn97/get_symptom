import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from fastapi import FastAPI, Query 
import uvicorn  
from typing import List  # List 타입을 위한 typing 모듈 가져오기

app = FastAPI()  # FastAPI 앱 인스턴스 생성

# 데이터 불러오기
kg3 = pd.read_csv('data/kg3.csv')
kg4 = pd.read_csv('data/kg4.csv')

# 첫 번째 페이지로 이동
@app.get("/")  # 루트 경로('/')에 대한 GET 요청 처리
async def read_item():
    return {"message": "Welcome to our app"}  # 환영 메시지 출력

@app.get("/symptom")  # '/symptom' 경로에 대한 GET 요청 처리
async def get_symptom_list(symptom_ids: List[int] = Query(...)):
    # Create a list of symptoms to be excluded
    excluded_symptoms = kg3.loc[kg3['y_index'].isin(symptom_ids), 'y_name'].unique()    

    # Create a list of symptoms to be excluded
    excluded_symptoms = kg3.loc[kg3['y_index'].isin(symptom_ids), 'y_name'].unique()

    # Filter kg3 dataframe by symptom_ids
    dat1 = kg3[kg3['y_index'].isin(symptom_ids)]
    # Perform a left join with the grouped and summarised version of dat1
    grouped_dat1 = dat1.groupby('x_index').size().reset_index(name='n_edge')
    dat1 = dat1.merge(grouped_dat1, on='x_index', how='left')

    # Identify the unique x_index values from dat1 where n_edge is not equal to 1
    unique_x_indices = dat1[dat1['n_edge'] != 1]['x_index'].unique()
    # Filter kg4 to include rows where x_index is in the unique_x_indices
    dat2 = kg4[kg4['x_index'].isin(unique_x_indices)].drop_duplicates()

    # Select x_name and y_name, and add a new column yn with all values as 1
    dat3 = dat2[['x_name', 'y_name']].copy()
    dat3['yn'] = 1
    # Remove duplicates
    dat3 = dat3.drop_duplicates()
    # Perform an anti-join
    grouped = dat2.groupby(['x_name', 'y_name']).size().reset_index(name='n')
    grouped_filtered = grouped[grouped['n'] > 1]
    dat3 = dat3.merge(grouped_filtered, on=['x_name', 'y_name'], how='left', indicator=True)
    dat3 = dat3[dat3['_merge'] == 'left_only'].drop(columns=['n', '_merge'])
    # Pivot the table
    dat3 = dat3.pivot_table(index='x_name', columns='y_name', values='yn', fill_value=0).reset_index()
    # Resetting the column names (optional)
    dat3.columns.name = None
    # Exclude the list of symptoms listed in the entry
    dat4 = dat3.loc[:, ~dat3.columns.isin(excluded_symptoms)]

    # Preparing the data for the decision tree model
    X = dat4.drop('x_name', axis=1)  # features
    y = dat4['x_name']  # target variable
    # Step 1: Create and fit the decision tree model
    model = DecisionTreeClassifier(criterion='entropy')
    model.fit(X, y)
    # Get feature importances
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)

    # Extract the name of the top feature as a string
    top_feature = feature_importances.idxmax()

    # Extract the corresponding y_index for the output, which is a top feature. 
    unique_index = kg4.loc[kg4['y_name'] == top_feature, 'y_index'].unique()
    unique_index = unique_index[0]

    return int(unique_index) # Return the index