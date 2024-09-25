import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

# CSV 파일 경로 설정
file_path = r'D:\128.치매 고위험군 라이프로그\01.데이터\1.Training\원천데이터\2.수면\train_sleep.csv' 
# CSV 파일을 판다스 데이터프레임으로 불러오기
Sleep = pd.read_csv(file_path)

# CSV 파일 경로 설정
file_path = r'D:\128.치매 고위험군 라이프로그\01.데이터\1.Training\라벨링데이터\2.수면\training_label.csv' 
# CSV 파일을 판다스 데이터프레임으로 불러오기
Sleep_label = pd.read_csv(file_path)


Sleep_label.rename(columns={'SAMPLE_EMAIL': 'EMAIL'}, inplace=True)
Sleep = pd.merge(Sleep, Sleep_label[['EMAIL', 'DIAG_NM']], on='EMAIL', how='left')
Sleep['EMAIL'] = Sleep['EMAIL'].str.extract(r'(\d{3})')

Sleep = Sleep.drop(['sleep_hr_5min', 'sleep_hypnogram_5min', 'sleep_rmssd_5min', 'sleep_is_longest',
                    'CONVERT(sleep_hr_5min USING utf8)',
                    'CONVERT(sleep_hypnogram_5min USING utf8)',
                    'CONVERT(sleep_rmssd_5min USING utf8)'
                    ], axis=1)

# 데이터프레임 출력
print(Sleep)
print(Sleep_label)

Sleep.rename(columns={
    'EMAIL': '이메일',
    'sleep_awake': '깨어있음',
    'sleep_bedtime_end': '종료시간',
    'sleep_bedtime_start': '시작시간',
    'sleep_breath_average': '평균호흡',
    'sleep_deep': '깊은수면',
    'sleep_duration': '총시간',
    'sleep_efficiency': '효율',
    'sleep_hr_average': '평균심박수',
    'sleep_hr_lowest': '최저심박수',
    'sleep_light': '얕은수면',
    'sleep_midpoint_at_delta': '중간점변화',
    'sleep_midpoint_time': '중간점시간',
    'sleep_onset_latency': '잠들기까지시간',
    'sleep_period_id': '기간_ID',
    'sleep_rem': '렘수면',
    'sleep_restless': '뒤척임',
    'sleep_rmssd': '심박변동성_RMSSD',
    'sleep_score': '점수',
    'sleep_score_alignment': '정렬점수',
    'sleep_score_deep': '깊은수면점수',
    'sleep_score_disturbances': '방해점수',
    'sleep_score_efficiency': '효율점수',
    'sleep_score_latency': '잠들기점수',
    'sleep_score_rem': '렘수면점수',
    'sleep_score_total': '총점',
    'sleep_temperature_delta': '체온변화',
    'sleep_temperature_deviation': '체온편차',
    'sleep_total': '총수면',
}, inplace=True)

# 컬럼 이름 확인
print(Sleep.columns)

# 불필요한 공백이 있을 경우 제거
Sleep.columns = Sleep.columns.str.strip()

Sleep['시작시간'] = pd.to_datetime(Sleep['시작시간'])
Sleep['종료시간'] = pd.to_datetime(Sleep['종료시간'])
Sleep['활동시간'] = Sleep['종료시간'] - Sleep['시작시간']

plt.rcParams['font.family'] = 'Malgun Gothic'  
plt.rcParams['axes.unicode_minus'] = False
Sleep['활동시간_분'] = Sleep['활동시간'].dt.total_seconds() / 60

plt.figure(figsize=(10, 6))
plt.hist(Sleep['활동시간_분'], bins=30, color='blue', alpha=0.7)
plt.title('활동시간 분포 (분)', fontsize=16)
plt.xlabel('활동시간 (분)', fontsize=14)
plt.ylabel('빈도수', fontsize=14)
plt.grid(True)
plt.show()

Sleep.drop(['시작시간', '종료시간','활동시간'], axis=1, inplace=True)
Sleep['DIAG_NM'] = Sleep['DIAG_NM'].replace({'CN': 0, 'MCI': 1, 'Dem': 2})
Sleep.drop(["이메일"], axis=1, inplace=True)


Sleep_data = Sleep.drop(columns=['DIAG_NM'])
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(Sleep_data)

Sleep_nom = pd.DataFrame(normalized_data, columns=Sleep_data.columns)
Sleep_nom['DIAG_NM'] = Sleep['DIAG_NM'].reset_index(drop=True)
print(Sleep_nom)

# 상관 행렬 계산
correlation_matrix = Sleep_nom.corr()

# 상관 행렬 시각화
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix of Sleep DataFrame')
plt.show()
