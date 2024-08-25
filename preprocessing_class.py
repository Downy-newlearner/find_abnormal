# 전처리 패키지
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA


# 모델링 패키지
from sklearn.ensemble import RandomForestClassifier
# from catboost import CatBoostClassifier
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier


# 모델 평가 및 검증 패키지
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score


# 기타 패키지
import random
import os
from datetime import datetime
import time
import math


# 전처리 클래스
# 입력은 train_begin, test_begin을 권장한다.
class Preprocessing:
    # 멤버 변수
    data = None # same_data를 저장하는 변수
    submission_diff = None 
    label_encoder = None
    X_train = None
    X_val = None
    y_train = None
    y_val = None
    X_test = None

    def __init__(self, train_data, test_data):
        # 1. 입력받은 train, test 데이터 합치기.(concat)
        # 2. Equipment(Line #1, #2)와 PalletID와 Production Qty가 모두 같은 데이터는 same_data에 저장, 그렇지 않은 데이터는 diff_data에 저장
        # 3. diff_data의 target이 nan인 데이터만 추출하여 diff_test에 저장 후 target을 'AbNormal'로 변경 -> submission_diff에 Set ID와 target만 저장
        # 4. same_data와 submission_diff 반환

        data = pd.concat([train_data, test_data], sort=False)
        data.reset_index(drop=True, inplace=True) # data의 인덱스를 0부터 순차적으로 변경(concat하면 index가 중복될 수 있기 때문에 reset_index() 사용)

        le = LabelEncoder()

        # 1) Equipment_Suffix 컬럼을 Label Encoding하기
        suffixes = ['Dam', 'Fill1', 'Fill2']
        for suffix in suffixes:
            for column in [f'Equipment_{suffix}']:
                data[column] = data[column].astype(str)
                le.fit(data[column])
                data[column] = le.transform(data[column])


        # 2) Model.Suffix_Dam, Workorder_Dam 컬럼을 Label Encoding하기
        li = ['Model.Suffix_Dam','Workorder_Dam']
        for column in li:
            data[column] = data[column].astype(str)
            le.fit(data[column])
            data[column] = le.transform(data[column])

        # 3) Chamber Temp. Judge Value_AutoClave(탈포 판단값) 컬럼을 Label Encoding하기
        data['Chamber Temp. Judge Value_AutoClave'] = data['Chamber Temp. Judge Value_AutoClave'].astype(str)
        le.fit(data['Chamber Temp. Judge Value_AutoClave'])
        data['Chamber Temp. Judge Value_AutoClave'] = le.transform(data['Chamber Temp. Judge Value_AutoClave'])

        # 4) GMES_ORIGIN_INSP_JUDGE_CODE Judge Value_AutoClave 컬럼을 Label Encoding하기
        data['GMES_ORIGIN_INSP_JUDGE_CODE Collect Result_AutoClave'] = data['GMES_ORIGIN_INSP_JUDGE_CODE Collect Result_AutoClave'].astype(str)
        le.fit(data['GMES_ORIGIN_INSP_JUDGE_CODE Collect Result_AutoClave'])
        data['GMES_ORIGIN_INSP_JUDGE_CODE Collect Result_AutoClave'] = le.transform(data['GMES_ORIGIN_INSP_JUDGE_CODE Collect Result_AutoClave'])
        data = data.drop(columns='GMES_ORIGIN_INSP_JUDGE_CODE Judge Value_AutoClave')

        # Equipment_Dam, Equipment_Fill1, Equipment_Fill2의 값을 비교하여 다르면 해당 데이터의 인덱스를 index_of_diff, 같으면 index_of_same 저장
        index_of_diff = []
        index_of_same = []

        dam_values = data['Equipment_Dam'].values
        fill1_values = data['Equipment_Fill1'].values
        fill2_values = data['Equipment_Fill2'].values

        index_of_diff = np.where((dam_values != fill1_values) | (dam_values != fill2_values))[0].tolist()
        index_of_same = np.where((dam_values == fill1_values) & (dam_values == fill2_values))[0].tolist()


        # PalletID Collect Result_Dam, PalletID Collect Result_Fill1, PalletID Collect Result_Dam 의 값을 비교하여 다르면 해당 데이터의 인덱스를 index_of_diff, 같으면 index_of_same append
        index_of_diff2 = []
        index_of_same2 = []

        dam_values = data['PalletID Collect Result_Dam'].values
        fill1_values = data['PalletID Collect Result_Fill1'].values
        fill2_values = data['PalletID Collect Result_Fill2'].values

        index_of_diff2 = np.where((dam_values != fill1_values) | (dam_values != fill2_values))[0].tolist()
        index_of_same2 = np.where((dam_values == fill1_values) & (dam_values == fill2_values))[0].tolist()


        # Production Qty Collect Result_Dam, Production Qty Collect Result_Fill1, Production Qty Collect Result_Fill2의 값을 비교하여 다르면 해당 데이터의 인덱스를 index_of_diff, 같으면 index_of_same append
        index_of_diff3 = []
        index_of_same3 = []

        dam_values = data['Production Qty Collect Result_Dam'].values
        fill1_values = data['Production Qty Collect Result_Fill1'].values
        fill2_values = data['Production Qty Collect Result_Fill2'].values

        index_of_diff3 = np.where((dam_values != fill1_values) | (dam_values != fill2_values))[0].tolist()
        index_of_same3 = np.where((dam_values == fill1_values) & (dam_values == fill2_values))[0].tolist()

        # 3개의 필터링 결과 합치기
        index_of_diff_total = list(set(index_of_diff + index_of_diff2 + index_of_diff3))

        # index_of_diff_total의 중복값 제거
        index_of_diff_total = list(set(index_of_diff_total))


        # index_of_diff_total에 해당하는 데이터를 diff_data에 저장하고, 해당하지 않는 나머지 데이터를 same_data에 저장
        diff_data = data.loc[index_of_diff_total]
        same_data = data.drop(index=index_of_diff_total)

        # diff_data의 'target' 컬럼이 nan인 행만 추출하여 diff_test에 저장
        diff_test = diff_data[diff_data['target'].isnull()]

        # diff_test의 'target' 컬럼의 값을 모두 'AbNormal'로 변경
        diff_test['target'] = 'AbNormal'

        # diff_test의 'Set ID'와 'target' 컬럼만 선택하여 submission_diff에 저장
        submission_diff = diff_test[['Set ID', 'target']].copy()

        self.data = same_data
        self.submission_diff = submission_diff
        self.label_encoder = LabelEncoder()

        
    
    def dam(self):
        print("preprocessing - dam")
        data = self.data.copy()
        columns1 = [
            'Stage1 Line1 Distance Speed Collect Result_Dam',
            'Stage1 Line2 Distance Speed Collect Result_Dam',
            'Stage1 Line3 Distance Speed Collect Result_Dam',
            'Stage1 Line4 Distance Speed Collect Result_Dam'
        ]

        # 주어진 칼럼들에서 편차를 계산하는 코드
        data['Stage1_Distance_Speed_StdDev'] = data[columns1].std(axis=1)

        columns2 = [
            'Stage2 Line1 Distance Speed Collect Result_Dam',
            'Stage2 Line2 Distance Speed Collect Result_Dam',
            'Stage2 Line3 Distance Speed Collect Result_Dam',
            'Stage2 Line4 Distance Speed Collect Result_Dam'
        ]

        # 주어진 칼럼들에서 편차를 계산하는 코드
        data['Stage2_Distance_Speed_StdDev'] = data[columns2].std(axis=1)

        columns3 = [
            'Stage3 Line1 Distance Speed Collect Result_Dam',
            'Stage3 Line2 Distance Speed Collect Result_Dam',
            'Stage3 Line3 Distance Speed Collect Result_Dam',
            'Stage3 Line4 Distance Speed Collect Result_Dam'
        ]

        # 주어진 칼럼들에서 편차를 계산하는 코드
        data['Stage3_Distance_Speed_StdDev'] = data[columns3].std(axis=1)


        # 열 목록 정의
        cols_to_average = ['Stage1 Circle1 Distance Speed Collect Result_Dam',
            'Stage1 Circle2 Distance Speed Collect Result_Dam',
            'Stage1 Circle3 Distance Speed Collect Result_Dam',
            'Stage1 Circle4 Distance Speed Collect Result_Dam',
            'Stage1 Line1 Distance Speed Collect Result_Dam',
            'Stage1 Line2 Distance Speed Collect Result_Dam',
            'Stage1 Line3 Distance Speed Collect Result_Dam',
            'Stage1 Line4 Distance Speed Collect Result_Dam',
            'Stage2 Circle1 Distance Speed Collect Result_Dam',
            'Stage2 Circle2 Distance Speed Collect Result_Dam',
            'Stage2 Circle3 Distance Speed Collect Result_Dam',
            'Stage2 Circle4 Distance Speed Collect Result_Dam',
            'Stage2 Line1 Distance Speed Collect Result_Dam',
            'Stage2 Line2 Distance Speed Collect Result_Dam',
            'Stage2 Line3 Distance Speed Collect Result_Dam',
            'Stage2 Line4 Distance Speed Collect Result_Dam',
            'Stage3 Circle1 Distance Speed Collect Result_Dam',
            'Stage3 Circle2 Distance Speed Collect Result_Dam',
            'Stage3 Circle3 Distance Speed Collect Result_Dam',
            'Stage3 Circle4 Distance Speed Collect Result_Dam',
            'Stage3 Line1 Distance Speed Collect Result_Dam',
            'Stage3 Line2 Distance Speed Collect Result_Dam',
            'Stage3 Line3 Distance Speed Collect Result_Dam',
            'Stage3 Line4 Distance Speed Collect Result_Dam',

        ]
        data['Average Stage1 CL Distance Speed Collect Result_Dam'] = data[cols_to_average[:8]].mean(axis=1)
        data['Average Stage2 CL Distance Speed Collect Result_Dam'] = data[cols_to_average[8:16]].mean(axis=1)
        data['Average Stage3 CL Distance Speed Collect Result_Dam'] = data[cols_to_average[16:]].mean(axis=1)
        data = data.drop(columns=cols_to_average)

        # 추가적인 상호작용 피처 생성
        # 속도와 시간의 조합으로 새로운 피처 생성 (예: 거리 계산)
        data['Speed_Time_Interaction_Stage1 Result_Dam'] = (
            data['DISCHARGED SPEED OF RESIN Collect Result_Dam'] * 
            data['DISCHARGED TIME OF RESIN(Stage1) Collect Result_Dam']
        )

        data['Speed_Time_Interaction_Stage2 Result_Dam'] = (
            data['DISCHARGED SPEED OF RESIN Collect Result_Dam'] * 
            data['DISCHARGED TIME OF RESIN(Stage2) Collect Result_Dam']
        )
        data['Speed_Time_Interaction_Stage3 Result_Dam'] = (
            data['DISCHARGED SPEED OF RESIN Collect Result_Dam'] * 
            data['DISCHARGED TIME OF RESIN(Stage3) Collect Result_Dam']
        )


        data['Total Speed_Time Result_Dam'] = data['Speed_Time_Interaction_Stage1 Result_Dam']+data['Speed_Time_Interaction_Stage2 Result_Dam']+data['Speed_Time_Interaction_Stage3 Result_Dam']

        # 도메인 지식을 활용한 피처 생성: 예를 들어 레진 분사 과정에서 발생할 수 있는 레진 양과 관련된 계산
        data['Total_Dispense_Volume Result_Dam'] = (
            data['Dispense Volume(Stage1) Collect Result_Dam'] +
            data['Dispense Volume(Stage2) Collect Result_Dam']+
            data['Dispense Volume(Stage3) Collect Result_Dam']
        )
        data['CURE POSITION X Collect Result_Dam'] = abs(data['CURE START POSITION X Collect Result_Dam']-data['CURE END POSITION X Collect Result_Dam'])
        data['CURE POSITION Z Collect Result_Dam'] = abs(data['CURE START POSITION Z Collect Result_Dam']-data['CURE END POSITION Z Collect Result_Dam'])
        data['CURE POSITION Θ Collect Result_Dam'] = abs(data['CURE START POSITION Θ Collect Result_Dam']-data['CURE END POSITION Θ Collect Result_Dam'])

        # data['CURE TIME X Collect Result_Dam'] = abs(data['CURE START POSITION X Collect Result_Dam']-data['CURE END POSITION X Collect Result_Dam'])/data['CURE SPEED Collect Result_Dam']
        # data['CURE TIME Z Collect Result_Dam'] = abs(data['CURE START POSITION Z Collect Result_Dam']-data['CURE END POSITION Z Collect Result_Dam'])/data['CURE SPEED Collect Result_Dam']
        # data['CURE TIME Θ Collect Result_Dam'] = abs(data['CURE START POSITION Θ Collect Result_Dam']-data['CURE END POSITION Θ Collect Result_Dam'])/data['CURE SPEED Collect Result_Dam']

        # a = abs(data['CURE START POSITION X Collect Result_Dam']-data['CURE END POSITION X Collect Result_Dam'])/data['CURE SPEED Collect Result_Dam']
        # b = abs(data['CURE START POSITION Z Collect Result_Dam']-data['CURE END POSITION Z Collect Result_Dam'])/data['CURE SPEED Collect Result_Dam']
        # c = abs(data['CURE START POSITION Θ Collect Result_Dam']-data['CURE END POSITION Θ Collect Result_Dam'])/data['CURE SPEED Collect Result_Dam']

        # data['CURE PT X Collect Result_Dam'] = data['CURE POSITION X Collect Result_Dam']*a
        # data['CURE PT Z Collect Result_Dam'] = data['CURE POSITION Z Collect Result_Dam']*b
        # data['CURE PT Θ Collect Result_Dam'] = data['CURE POSITION Θ Collect Result_Dam']*c

        data['CURE DT X Collect Result_Dam'] = abs(data['CURE START POSITION X Collect Result_Dam']-data['CURE END POSITION X Collect Result_Dam'])*data['CURE SPEED Collect Result_Dam']
        data['CURE DT Z Collect Result_Dam'] = abs(data['CURE START POSITION Z Collect Result_Dam']-data['CURE END POSITION Z Collect Result_Dam'])*data['CURE SPEED Collect Result_Dam']
        data['CURE DT Θ Collect Result_Dam'] = abs(data['CURE START POSITION Θ Collect Result_Dam']-data['CURE END POSITION Θ Collect Result_Dam'])*data['CURE SPEED Collect Result_Dam']

        # 기존 열 삭제
        data = data.drop(columns=[
                                'CURE START POSITION X Collect Result_Dam','CURE START POSITION Θ Collect Result_Dam',
                                'CURE END POSITION X Collect Result_Dam','CURE END POSITION Z Collect Result_Dam','CURE END POSITION Θ Collect Result_Dam',
                                'CURE SPEED Collect Result_Dam',
                                        ])

        data['THICKNESS_Range_Collect_Result_Dam'] = (
            data[['THICKNESS 1 Collect Result_Dam', 'THICKNESS 2 Collect Result_Dam', 'THICKNESS 3 Collect Result_Dam']].max(axis=1) -
            data[['THICKNESS 1 Collect Result_Dam', 'THICKNESS 2 Collect Result_Dam', 'THICKNESS 3 Collect Result_Dam']].min(axis=1)
                )
        # data['THICKNESS_Avg_Collect_Result_Dam'] = (
        #     data['THICKNESS 1 Collect Result_Dam'] +
        #     data['THICKNESS 2 Collect Result_Dam'] +
        #     data['THICKNESS 3 Collect Result_Dam']
        # ) / 3

        # data['THICKNESS_StdDev_Collect_Result_Dam'] = data[
        #     ['THICKNESS 1 Collect Result_Dam', 'THICKNESS 2 Collect Result_Dam', 'THICKNESS 3 Collect Result_Dam']
        # ].std(axis=1)

        self.data = data
            

    def fill1(self):
        print("preprocessing - fill1")

        data = self.data.copy()

        # 제거할 변수
        drop_cols = [
            'Equipment_Fill1',
            'Model.Suffix_Fill1',
            'Workorder_Fill1',
            'PalletID Collect Result_Fill1',
            'Production Qty Collect Result_Fill1',
            'Receip No Collect Result_Fill1'
        ]
        
        # 각 Stage에서의 DISCHARGED 양 계산
        # data['DISCHARGED_AMOUNT_STAGE1'] = data['DISCHARGED SPEED OF RESIN Collect Result_Fill1'] * data['DISCHARGED TIME OF RESIN(Stage1) Collect Result_Fill1']
        # data['DISCHARGED_AMOUNT_STAGE2'] = data['DISCHARGED SPEED OF RESIN Collect Result_Fill1'] * data['DISCHARGED TIME OF RESIN(Stage2) Collect Result_Fill1']
        # data['DISCHARGED_AMOUNT_STAGE3'] = data['DISCHARGED SPEED OF RESIN Collect Result_Fill1'] * data['DISCHARGED TIME OF RESIN(Stage3) Collect Result_Fill1']

        # Dispense 양과 DISCHARGED 양의 상관계수 계산
        # Stage 1 Correlation: 0.9795678484346679
        # Stage 2 Correlation: 0.9986658296558463
        # Stage 3 Correlation: 0.9571706530033417
        
        # Dispense Volume Collect Result_Fill1 만 사용하기로 결정
        data['Dispense Volume Collect Result_Fill1'] = list(zip(data['Dispense Volume(Stage1) Collect Result_Fill1'],
                                        data['Dispense Volume(Stage2) Collect Result_Fill1'],
                                        data['Dispense Volume(Stage3) Collect Result_Fill1']))
        data = data.drop(columns=[
            'Dispense Volume(Stage1) Collect Result_Fill1',
            'Dispense Volume(Stage2) Collect Result_Fill1',
            'Dispense Volume(Stage3) Collect Result_Fill1'], errors='ignore')
        
        # DISCHARGED와 SPEED 삭제
        data = data.drop(columns=[
            'DISCHARGED SPEED OF RESIN Collect Result_Fill1',
            'DISCHARGED TIME OF RESIN(Stage1) Collect Result_Fill1',
            'DISCHARGED TIME OF RESIN(Stage2) Collect Result_Fill1',
            'DISCHARGED TIME OF RESIN(Stage3) Collect Result_Fill1'], errors='ignore')
        
        # HEAD NORMAL COORDINATE X AXIS(1, 2, 3)
        data['HEAD NORMAL COORDINATE X AXIS(1, 2, 3)'] = list(zip(data['HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Fill1'],
                                        data['HEAD NORMAL COORDINATE X AXIS(Stage2) Collect Result_Fill1'],
                                        data['HEAD NORMAL COORDINATE X AXIS(Stage3) Collect Result_Fill1']))
        
        # HEAD NORMAL COORDINATE Y AXIS(1, 2, 3)
        """
        data['HEAD NORMAL COORDINATE Y AXIS(1, 2, 3)'] = list(zip(data['HEAD NORMAL COORDINATE Y AXIS(Stage1) Collect Result_Fill1'],
                                        data['HEAD NORMAL COORDINATE Y AXIS(Stage2) Collect Result_Fill1'],
                                        data['HEAD NORMAL COORDINATE Y AXIS(Stage3) Collect Result_Fill1']))
        """
                
        data['HEAD NORMAL COORDINATE Y AXIS(1, 2, 3)'] = (
            (data['HEAD NORMAL COORDINATE Y AXIS(Stage1) Collect Result_Fill1'] + data['HEAD NORMAL COORDINATE Y AXIS(Stage2) Collect Result_Fill1'] + data['HEAD NORMAL COORDINATE Y AXIS(Stage3) Collect Result_Fill1']) / 3
        )
        
        # HEAD NORMAL COORDINATE Z AXIS(1, 2, 3)
        """
        data['HEAD NORMAL COORDINATE Z AXIS(1, 2, 3)'] = list(zip(data['HEAD NORMAL COORDINATE Z AXIS(Stage1) Collect Result_Fill1'],
                                        data['HEAD NORMAL COORDINATE Z AXIS(Stage2) Collect Result_Fill1'],
                                        data['HEAD NORMAL COORDINATE Z AXIS(Stage3) Collect Result_Fill1']))
        """
        
        data['HEAD NORMAL COORDINATE Z AXIS(1, 2, 3)'] = (
            (data['HEAD NORMAL COORDINATE Z AXIS(Stage1) Collect Result_Fill1'] + data['HEAD NORMAL COORDINATE Z AXIS(Stage2) Collect Result_Fill1'] + data['HEAD NORMAL COORDINATE Z AXIS(Stage3) Collect Result_Fill1']) / 3
        )

        
        # 사용한 칼럼 제거
        data = data.drop(columns=[
            'HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Fill1',
            'HEAD NORMAL COORDINATE X AXIS(Stage2) Collect Result_Fill1',
            'HEAD NORMAL COORDINATE X AXIS(Stage3) Collect Result_Fill1',
            'HEAD NORMAL COORDINATE Y AXIS(Stage1) Collect Result_Fill1',
            'HEAD NORMAL COORDINATE Y AXIS(Stage2) Collect Result_Fill1',
            'HEAD NORMAL COORDINATE Y AXIS(Stage3) Collect Result_Fill1',
            'HEAD NORMAL COORDINATE Z AXIS(Stage1) Collect Result_Fill1',
            'HEAD NORMAL COORDINATE Z AXIS(Stage2) Collect Result_Fill1',
            'HEAD NORMAL COORDINATE Z AXIS(Stage3) Collect Result_Fill1'], errors='ignore')
        
        # drop_cols 제거
        data = data.drop(columns=drop_cols, errors='ignore')
        
        self.data = data
        
    
    def fill2(self):
        print("preprocessing - fill2")
        data = self.data.copy()


        # 제거할 변수 이름 
        drop_cols = [
            'Equipment_Fill2',
            'Model.Suffix_Fill2',
            'Workorder_Fill2',
            'CURE STANDBY POSITION Z Collect Result_Fill2',
            'PalletID Collect Result_Fill2',
            'Production Qty Collect Result_Fill2',
            'Receip No Collect Result_Fill2'
        ]

        # CURE TRACK POSITION X_Fill2
        conditions_CURE_X = [
            (data['CURE START POSITION X Collect Result_Fill2'] == 1020) & (data['CURE END POSITION X Collect Result_Fill2'] == 240),
            (data['CURE START POSITION X Collect Result_Fill2'] == 240) & (data['CURE END POSITION X Collect Result_Fill2'] == 1020)]

        choices_CURE_X = ['1020_to_240', '240_to_1020']

        data['CURE TRACK POSITION X_Fill2'] = np.select(conditions_CURE_X, choices_CURE_X, default='Unknown') # 아무것도 해당하지 않는 경우 Unknown 값이 할당

        data = data.drop(columns=[
            'CURE START POSITION X Collect Result_Fill2',
            'CURE END POSITION X Collect Result_Fill2'], errors='ignore')

        # CURE TRACK POSITION Z_Fill2
        
        # HEAD NORMAL COORDINATE X AXIS_Fill2

        data['HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Fill2'].replace(304.8, 305.0, inplace=True) # 304.8을 305.0으로 대체
        data['HEAD NORMAL COORDINATE X AXIS(Stage3) Collect Result_Fill2'].replace(692.8, 694.0, inplace=True) # 692.8을 694.0으로 대체
        
        conditions_HEAD_X = [
            (data['HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Fill2'] == 835.5) & 
            (data['HEAD NORMAL COORDINATE X AXIS(Stage2) Collect Result_Fill2'] == 458.0) & 
            (data['HEAD NORMAL COORDINATE X AXIS(Stage3) Collect Result_Fill2'] == 156.0),
    
            (data['HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Fill2'] == 305.0) & 
            (data['HEAD NORMAL COORDINATE X AXIS(Stage2) Collect Result_Fill2'] == 499.8) & 
            (data['HEAD NORMAL COORDINATE X AXIS(Stage3) Collect Result_Fill2'] == 694.0)]

        choices_HEAD_X = ['835.5_458.0_156.0', '305.0_499.8_694.0']

        data['HEAD NORMAL COORDINATE X AXIS_Fill2'] = np.select(conditions_HEAD_X, choices_HEAD_X, default='Unknown') # 아무것도 해당하지 않는 경우 Unknown 값이 할당

        data = data.drop(columns=[
            'HEAD NORMAL COORDINATE X AXIS(Stage1) Collect Result_Fill2', 
            'HEAD NORMAL COORDINATE X AXIS(Stage2) Collect Result_Fill2', 
            'HEAD NORMAL COORDINATE X AXIS(Stage3) Collect Result_Fill2'], errors='ignore')

        # HEAD NORMAL COORDINATE Y AXIS_Fill2
        data['HEAD NORMAL COORDINATE Y AXIS_Fill2'] = data['HEAD NORMAL COORDINATE Y AXIS(Stage1) Collect Result_Fill2']

        data = data.drop(columns=[
            'HEAD NORMAL COORDINATE Y AXIS(Stage1) Collect Result_Fill2', 
            'HEAD NORMAL COORDINATE Y AXIS(Stage2) Collect Result_Fill2', 
            'HEAD NORMAL COORDINATE Y AXIS(Stage3) Collect Result_Fill2'], errors='ignore')

        # HEAD NORMAL COORDINATE Z AXIS_Fill2
        data['HEAD NORMAL COORDINATE Z AXIS_Fill2'] = data['HEAD NORMAL COORDINATE Z AXIS(Stage1) Collect Result_Fill2']

        data = data.drop(columns=[
            'HEAD NORMAL COORDINATE Z AXIS(Stage1) Collect Result_Fill2',
            'HEAD NORMAL COORDINATE Z AXIS(Stage2) Collect Result_Fill2',
            'HEAD NORMAL COORDINATE Z AXIS(Stage3) Collect Result_Fill2'], errors='ignore')

        # drop_cols 제거
        data = data.drop(columns=drop_cols, errors='ignore')


        self.data = data

    

    def autoclave(self):
        print("preprocessing - autoclave")
        data = self.data.copy()
        
        # Model.Suffix_AutoClave, Workorder_AutoClave 컬럼 제거
        data = data.drop(columns=['Model.Suffix_AutoClave', 'Workorder_AutoClave'])

        # Pressure Amount 컬럼 만들기
        pressure_amount = data['1st Pressure Collect Result_AutoClave'] * data['1st Pressure 1st Pressure Unit Time_AutoClave']
        pressure_amount += data['2nd Pressure Collect Result_AutoClave'] * data['2nd Pressure Unit Time_AutoClave']
        pressure_amount += data['3rd Pressure Collect Result_AutoClave'] * data['3rd Pressure Unit Time_AutoClave']

        self.data = data
       

    def final_preprocessing(self):
        print("final preprocessing")
        all_data = self.data.copy()
        print(f"all_data shape: {all_data.shape}")

        # 불필요한 칼럼 제거
        unnecessary_columns =['Model.Suffix_AutoClave','Model.Suffix_Fill1','Model.Suffix_Fill2',
            'Workorder_AutoClave','Workorder_Fill1','Workorder_Fill2',
            'Receip No Collect Result_Fill1','Receip No Collect Result_Fill2',]
        
        
        for column in unnecessary_columns:
            if column in all_data.columns:
                all_data.drop(columns=[column], inplace=True)

        print("section 1")
        # 1. 비대칭성이 높은 칼럼에 대한 정규화(로그 변환) 및 표준화
        a=[]
        for c in all_data.columns:
            if all_data[c].nunique()==1:
                a.append(c)
        all_data.drop(columns=a, inplace=True)

        # 모든 칼럼의 왜도(Skewness) 계산
        train_data = all_data[pd.notnull(all_data['target'])]
        X = train_data.drop(columns=['target','Set ID'])

        # train_data의 컬럼 중 object형 컬럼을 제외한 컬럼만 선택
        X = X.select_dtypes(exclude='object')

        for col in X.columns:
            # 각 컬럼의 dtype 출력
            print(f'{col} dtype: {X[col].dtype}')
        skewness = X.skew().sort_values(ascending=False)

        # 비대칭성이 높은 칼럼(왜도의 절대값이 1보다 큰 칼럼) 추출
        high_skew_cols_list = skewness[abs(skewness) > 1].index.tolist()

        # 로그 변환을 위한 작은 상수 추가
        epsilon = 1e-6

        # 로그 변환을 적용할 칼럼 리스트에서 'PalletID'와 'Production Qty'를 제외
        high_skew_cols_list = [col for col in high_skew_cols_list if col not in ['PalletID', 'Production Qty']]

        # 로그 변환 적용
        for col in high_skew_cols_list:
            if any(all_data[col] <= 0):  # 0 또는 음수 값이 있는 경우 상수 추가
                all_data[col] = np.log1p(all_data[col] + epsilon)
            else:
                all_data[col] = np.log1p(all_data[col])

        print("section 2")

        # 표준화
        scaler = StandardScaler()
        all_data[high_skew_cols_list] = scaler.fit_transform(all_data[high_skew_cols_list])




        print("section 3")

        # 2. LabelEncoder 초기화
        le = LabelEncoder()

        train_data = all_data[pd.notnull(all_data['target'])] # train 데이터 추출: 'target' 컬럼이 NaN이 아닌 데이터
        print("train_data shape: ", train_data.shape)
        # target과 feature 분리
        train_data['target'] = train_data['target'].astype(str)  # target 변수
        train_data['target'] = le.fit_transform(train_data['target'])  # target 변수를 숫자로 변환

        # train_data에서 object형 컬럼에 대한 Label Encoding
        for column in train_data.select_dtypes(include='object').columns:
            train_data[column] = train_data[column].astype(str)
            le.fit(train_data[column])
            train_data[column] = le.transform(train_data[column])

        print("section 4")
        print("train_data shape111: ", train_data.shape)

        # 3. PCA 적용
        # train_data - (target 드랍) -> features - (정규화) -> scaled_features - (PCA 적용) -> pca_features
        train_data = train_data.drop(columns=['Set ID'])
        test_data = all_data[pd.isnull(all_data['target'])]
        test_x = test_data.drop(columns=['target'])
        # all_data에서 'target'이 nan인 데이터만 추출하여 test_data에 저장
        test_Set_ID = test_x['Set ID']

        # test_x에서 object형 컬럼에 대한 Label Encoding
        for column in test_x.select_dtypes(include='object').columns:
            test_x[column] = test_x[column].astype(str)
            le.fit(test_x[column])
            test_x[column] = le.transform(test_x[column])
        print("train_data shape222: ", train_data.shape)


        # 타겟 변수를 제외한 피처들만 사용
        features = train_data.drop(columns=['target', 'HEAD NORMAL COORDINATE X AXIS(1, 2, 3)', 'Dispense Volume Collect Result_Fill1'])

        print("features shape: ", features.shape)

        temp_drop = train_data[['HEAD NORMAL COORDINATE X AXIS(1, 2, 3)', 'Dispense Volume Collect Result_Fill1']]
        test_x_before_concat = test_x.drop(columns=['HEAD NORMAL COORDINATE X AXIS(1, 2, 3)', 'Dispense Volume Collect Result_Fill1', 'Set ID'])
        test_x_temp = test_x[['HEAD NORMAL COORDINATE X AXIS(1, 2, 3)', 'Dispense Volume Collect Result_Fill1']]
        
        # features 데이터프레임에서 수치형 컬럼에 대해서만 데이터 정규화
        scaler = StandardScaler()
        scaled_features_train = scaler.fit_transform(features)

        print("scaled_features_train shape111: ", scaled_features_train.shape)

        scaled_features_train = pd.DataFrame(scaled_features_train, columns=features.columns)



        scaled_features_test = scaler.transform(test_x_before_concat)
        scaled_features_test = pd.DataFrame(scaled_features_test, columns=test_x_before_concat.columns)


        print("scaled_features_train shape222: ", scaled_features_train.shape)

        X_test = pd.concat([test_x_temp.reset_index(drop=True),scaled_features_test], axis=1)
        scaled_features_train = pd.concat([scaled_features_train, temp_drop.reset_index(drop=True)], axis=1)

        print("temp_drop shape333: ", temp_drop.shape)


        print("scaled_features_train shape333: ", scaled_features_train.shape)

        # scaled_features_train, X_test의 컬럼 순서 맞추기
        print("X_test shape before: ", X_test.shape)

        X_test = X_test[scaled_features_train.columns]
        print("X_test shape after: ", X_test.shape)

        self.X_test = X_test
        self.Set_ID = test_Set_ID

        print("section 5")

        # 4. SMOTE 적용 및 train_test_split
        target = train_data['target']

        target = target.astype(str)
        target = le.fit_transform(target)

        # 데이터 분리 (학습용 80%, 검증용 20%)
        print("scaled_features_train shape: ", scaled_features_train.shape)
        X_train, X_val, y_train, y_val = train_test_split(scaled_features_train, target, test_size=0.2, random_state=42)

        # SMOTE 적용
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        self.X_train = X_train_smote
        self.X_val = X_val
        self.y_train = y_train_smote
        self.y_val = y_val


    def final_preprocessing_v2(self):
        print("Final preprocessing")
        all_data = self.data.copy()
        
        # 불필요한 칼럼 제거
        unnecessary_columns = [
            'Model.Suffix_AutoClave', 'Model.Suffix_Fill1', 'Model.Suffix_Fill2',
            'Workorder_AutoClave', 'Workorder_Fill1', 'Workorder_Fill2',
            'Receip No Collect Result_Fill1', 'Receip No Collect Result_Fill2'
        ]
        
        all_data.drop(columns=[col for col in unnecessary_columns if col in all_data.columns], inplace=True)
        
        print("Section 1")
        # 1. 고유 값이 1인 칼럼 제거
        all_data.drop(columns=[c for c in all_data if all_data[c].nunique() == 1], inplace=True)

        # 모든 칼럼의 왜도(Skewness) 계산
        train_data = all_data[all_data['target'].notnull()]
        X = train_data.drop(columns=['target', 'Set ID']).select_dtypes(exclude='object')

        # 왜도 계산 및 비대칭성이 높은 칼럼 추출
        skewness = X.skew().sort_values(ascending=False)
        high_skew_cols_list = skewness[abs(skewness) > 1].index.tolist()
        
        # 로그 변환을 위한 상수 추가
        epsilon = 1e-6
        high_skew_cols_list = [col for col in high_skew_cols_list if col not in ['PalletID', 'Production Qty']]
        
        # 로그 변환 적용
        for col in high_skew_cols_list:
            all_data[col] = np.log1p(all_data[col] + epsilon) if any(all_data[col] <= 0) else np.log1p(all_data[col])

        print("Section 2")

        # 표준화
        scaler = StandardScaler()
        all_data[high_skew_cols_list] = scaler.fit_transform(all_data[high_skew_cols_list])

        print("Section 3")

        # LabelEncoder 초기화 및 타겟 변환
        le = LabelEncoder()
        train_data['target'] = le.fit_transform(train_data['target'].astype(str))
        
        # 객체형 컬럼에 대한 Label Encoding
        for column in train_data.select_dtypes(include='object').columns:
            train_data[column] = le.fit_transform(train_data[column].astype(str))

        print("Section 4")
        print("Train data shape: ", train_data.shape)

        # PCA 적용 준비
        train_data = train_data.drop(columns=['Set ID'])
        test_data = all_data[all_data['target'].isnull()]
        test_x = test_data.drop(columns=['target'])

        # test_x에서 객체형 컬럼에 대한 Label Encoding
        for column in test_x.select_dtypes(include='object').columns:
            test_x[column] = le.fit_transform(test_x[column].astype(str))

        features = train_data.drop(columns=['target', 'HEAD NORMAL COORDINATE X AXIS(1, 2, 3)', 'Dispense Volume Collect Result_Fill1'])
        temp_drop = train_data[['HEAD NORMAL COORDINATE X AXIS(1, 2, 3)', 'Dispense Volume Collect Result_Fill1']]
        
        print("Features shape: ", features.shape)

        # 수치형 컬럼 대해 데이터 정규화
        scaled_features_train = pd.DataFrame(StandardScaler().fit_transform(features), columns=features.columns)
        scaled_features_test = pd.DataFrame(StandardScaler().fit_transform(test_x.drop(columns=['HEAD NORMAL COORDINATE X AXIS(1, 2, 3)', 'Dispense Volume Collect Result_Fill1', 'Set ID'])),
                                            columns=test_x.drop(columns=['HEAD NORMAL COORDINATE X AXIS(1, 2, 3)', 'Dispense Volume Collect Result_Fill1', 'Set ID']).columns)

        # X_test 생성 및 컬럼 순서 맞추기
        X_test = pd.concat([test_x[['HEAD NORMAL COORDINATE X AXIS(1, 2, 3)', 'Dispense Volume Collect Result_Fill1']].reset_index(drop=True), scaled_features_test.reset_index(drop=True)], axis=1)
        scaled_features_train = pd.concat([scaled_features_train, temp_drop.reset_index(drop=True)], axis=1)

        # X_test의 컬럼 순서 맞추기
        X_test = X_test[scaled_features_train.columns]

        self.X_test = X_test
        self.Set_ID = test_x['Set ID']

        print("Section 5")

        # SMOTE 적용 및 train_test_split
        target = train_data['target']
        X_train, X_val, y_train, y_val = train_test_split(scaled_features_train, target, test_size=0.2, random_state=42)

        # SMOTE 적용
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(X_train, y_train)
        self.X_val = X_val
        self.y_val = y_val