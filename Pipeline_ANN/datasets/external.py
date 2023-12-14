import pandas as pd

def external_data():
    light_df = pd.read_csv('../data/external/대구 보안등 정보.csv', encoding='cp949')[['설치개수', '소재지지번주소']]
    child_area_df = pd.read_csv('../data/external/대구 어린이 보호 구역 정보.csv', encoding='cp949').drop_duplicates()[['소재지지번주소']]
    parking_df = pd.read_csv('../data/external/대구 주차장 정보.csv', encoding='cp949')[['소재지지번주소', '급지구분']]
    
    child_area_df['cnt'] = 1
    parking_df = pd.get_dummies(parking_df, columns=['급지구분'])
    
    location_pattern = r'(\S+) (\S+) (\S+) (\S+)'

    light_df[['도시', '구', '동', '번지']] = light_df['소재지지번주소'].str.extract(location_pattern)
    light_df = light_df.drop(columns=['소재지지번주소', '번지'])

    light_df = light_df.groupby(['도시', '구', '동']).sum().reset_index()
    light_df.reset_index(inplace=True, drop=True)

    location_pattern = r'(\S+) (\S+) (\S+) (\S+)'

    child_area_df[['도시', '구', '동', '번지']] = child_area_df['소재지지번주소'].str.extract(location_pattern)
    child_area_df = child_area_df.drop(columns=['소재지지번주소', '번지'])

    child_area_df = child_area_df.groupby(['도시', '구', '동']).sum().reset_index()
    child_area_df.reset_index(inplace=True, drop=True)

    location_pattern = r'(\S+) (\S+) (\S+) (\S+)'

    parking_df[['도시', '구', '동', '번지']] = parking_df['소재지지번주소'].str.extract(location_pattern)
    parking_df = parking_df.drop(columns=['소재지지번주소', '번지'])

    parking_df = parking_df.groupby(['도시', '구', '동']).sum().reset_index()
    parking_df.reset_index(inplace=True, drop=True)
    
    ex_df = pd.merge(light_df, child_area_df, how='left', on=['도시', '구', '동'])
    ex_df = pd.merge(ex_df, parking_df, how='left', on=['도시', '구', '동'])
       
    return ex_df