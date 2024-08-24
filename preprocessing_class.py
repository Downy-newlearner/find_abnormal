import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 전처리 클래스
# 입력은 train_begin, test_begin을 권장한다.
class Preprocessing:
    # 멤버 변수
    data = None # same_data를 저장하는 변수
    submission_diff = None 
    label_encoder = None


    def __init__(self, train_data, test_data):
        # 1. 입력받은 train, test 데이터 합치기.(concat)
        # 2. Equipment(Line #1, #2)와 PalletID와 Production Qty가 모두 같은 데이터는 same_data에 저장, 그렇지 않은 데이터는 diff_data에 저장
        # 3. diff_data의 target이 nan인 데이터만 추출하여 diff_test에 저장 후 target을 'AbNormal'로 변경 -> submission_diff에 Set ID와 target만 저장
        # 4. same_data와 submission_diff 반환


        data = pd.concat([train_data, test_data], sort=False)
        data.reset_index(drop=True, inplace=True) # data의 인덱스를 0부터 순차적으로 변경(concat하면 index가 중복될 수 있기 때문에 reset_index() 사용)

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

        
    
    def dam(self, data):
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

        ]
        data['Average Stage1 CL Distance Speed Collect Result_Dam'] = data[cols_to_average[:8]].mean(axis=1)
        data['Average Stage2 CL Distance Speed Collect Result_Dam'] = data[cols_to_average[8:]].mean(axis=1)
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


        data['Total Speed_Time Result_Dam'] = data['Speed_Time_Interaction_Stage1 Result_Dam']+data['Speed_Time_Interaction_Stage2 Result_Dam']

        # 도메인 지식을 활용한 피처 생성: 예를 들어 레진 분사 과정에서 발생할 수 있는 레진 양과 관련된 계산
        data['Total_Dispense_Volume Result_Dam'] = (
            data['Dispense Volume(Stage1) Collect Result_Dam'] +
            data['Dispense Volume(Stage2) Collect Result_Dam']
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

        return data #전처리 된 데이터 반환
            

    def fill1(self, data):
        #

        return data #전처리 된 데이터 반환
    
    
    def fill2(self, data):

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


        return data #전처리 된 데이터 반환
    

    def autoclave(self, data):
        
        # Model.Suffix_AutoClave, Workorder_AutoClave 컬럼 제거
        data = data.drop(columns=['Model.Suffix_AutoClave', 'Workorder_AutoClave'])

        # Pressure Amount 컬럼 만들기
        pressure_amount = data['1st Pressure Collect Result_AutoClave'] * data['1st Pressure 1st Pressure Unit Time_AutoClave']
        pressure_amount += data['2nd Pressure Collect Result_AutoClave'] * data['2nd Pressure Unit Time_AutoClave']
        pressure_amount += data['3rd Pressure Collect Result_AutoClave'] * data['3rd Pressure Unit Time_AutoClave']

        return data #전처리 된 데이터 반환
