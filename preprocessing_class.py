# 전처리 클래스 만들기
# 멤버 변수로 데이터프레임을 받아서 전처리를 수행하는 클래스를 만들어보자

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class Preprocessing:
    def __init__(self, data):
        self.data = data
        self.label_encoder = LabelEncoder()
        
    
    def distingush_always_AbNormal(self, data):
        


        return data_same, data_diff #전처리 된 데이터 반환
    
    def dam(self, data):
        

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
        #

        return data #전처리 된 데이터 반환
