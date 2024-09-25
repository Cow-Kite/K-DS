import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

#인지 정상(CN), 경도 인지 장애(MCI) 및 치매(DEM)
file_path = "1.Training/원천데이터/1.걸음걸이/train_activity.csv"
Walk = pd.read_csv(file_path)
file_path = "1.Training/라벨링데이터/1.걸음걸이/training_label.csv"
Walk_label = pd.read_csv(file_path)
Walk_label.rename(columns={'SAMPLE_EMAIL': 'EMAIL'}, inplace=True)
Walk = pd.merge(Walk, Walk_label[['EMAIL', 'DIAG_NM']], on='EMAIL', how='left')
Walk['EMAIL'] = Walk['EMAIL'].str.extract(r'(\d{3})')
#MET =휴식하고 있을 때 필요한 에너지나 몸에서 필요로 하는 산소의 양
Walk.rename(columns={
    'activity_average_met': '하루간 평균 MET',
    'activity_cal_active': '하루간 활동 칼로리',
    'activity_cal_total': '하루간 총 사용 칼로리',
    'activity_class_5min': '하루간 5분당 활동 로그',
    'activity_daily_movement': '매일 움직인 거리',
    'activity_day_end': '활동 종료 시간',
    'activity_day_start': '활동 시작 시간',
    'activity_high': '고강도 활동 시간',
    'activity_inactive': '비활동 시간',
    'activity_inactivity_alerts': '비활동 알람 횟수',
    'activity_low': '저강도 활동 시간',
    'activity_medium': '중강도 활동 시간',
    'activity_met_1min': '하루간 1분 당 MET 로그',
    'activity_met_min_high': '하루간 고강도 활동 MET',
    'activity_met_min_inactive': '하루간 비활동 MET',
    'activity_met_min_low': '하루간 저강도 활동 MET',
    'activity_met_min_medium': '하루간 중강도 활동 MET',
    'activity_non_wear': '미착용 시간',
    'activity_rest': '휴식 시간',
    'activity_score': '활동 점수',
    'activity_score_meet_daily_targets': '활동 목표달성 점수',
    'activity_score_move_every_hour': '매 시간 당 활동유지 점수',
    'activity_score_recovery_time': '회복시간 점수',
    'activity_score_stay_active': '활동 유지 점수',
    'activity_score_training_frequency': '운동 빈도 점수',
    'activity_score_training_volume': '운동 점수',
    'activity_steps': '매일 걸음 수',
    'activity_total': '활동 총 시간(분)'
}, inplace=True)
Walk['활동 시작 시간'] = pd.to_datetime(Walk['활동 시작 시간'])
Walk['활동 종료 시간'] = pd.to_datetime(Walk['활동 종료 시간'])
Walk['활동시간'] = Walk['활동 종료 시간'] - Walk['활동 시작 시간']

plt.rcParams['font.family'] = 'Malgun Gothic'  
plt.rcParams['axes.unicode_minus'] = False
Walk['활동시간_분'] = Walk['활동시간'].dt.total_seconds() / 60

plt.figure(figsize=(10, 6))
plt.hist(Walk['활동시간_분'], bins=30, color='blue', alpha=0.7)
plt.title('활동시간 분포 (분)', fontsize=16)
plt.xlabel('활동시간 (분)', fontsize=14)
plt.ylabel('빈도수', fontsize=14)
plt.grid(True)
plt.show()
Walk.drop(['활동 시작 시간', '활동 종료 시간','활동시간',"활동시간_분"], axis=1, inplace=True)
Walk['하루간 5분당 활동 로그'] = Walk['CONVERT(activity_class_5min USING utf8)'].apply(lambda x: len(x.split('/')))
Walk['하루간 1분 당 MET 로그'] = Walk['CONVERT(activity_met_1min USING utf8)'].apply(lambda x: len(str(x).split('/')))
Walk.drop(['CONVERT(activity_class_5min USING utf8)','CONVERT(activity_met_1min USING utf8)'], axis=1, inplace=True)
Walk['DIAG_NM'] = Walk['DIAG_NM'].replace({'CN': 0, 'MCI': 1, 'Dem': 2})
Walk.drop(["EMAIL"], axis=1, inplace=True)



walk_data = Walk.drop(columns=['DIAG_NM'])
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(walk_data)

Walk_nom = pd.DataFrame(normalized_data, columns=walk_data.columns)
Walk_nom['DIAG_NM'] = Walk['DIAG_NM'].reset_index(drop=True)

# 상관 행렬 계산
correlation_matrix = Walk_nom.corr()

# 상관 행렬 시각화
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix of Walk DataFrame')
plt.show()

