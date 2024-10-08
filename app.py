import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import association_rules, apriori
import streamlit as st

st.set_page_config(page_title='Market Basket Analysis')
st.title('Market Basket Analysis with Apriori Algorithm')

# Data loading
@st.cache_data
def load_data():
    return pd.read_csv('bread basket.csv')

df = load_data()

# Data preprocessing
df['date_time'] = pd.to_datetime(df['date_time'], format='%d-%m-%Y %H:%M')
df['month'] = df['date_time'].dt.month
df['day'] = df['date_time'].dt.weekday
df['hour'] = df['date_time'].dt.hour

df['month'] = df['month'].replace([i for i in range(1, 13)], 
                                   ["January", "February", "March", "April", "May", "June", 
                                    "July", "August", "September", "October", "November", 
                                    "December"])

df['day'] = df['day'].replace([i for i in range(0, 7)], 
                               ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", 
                                "Saturday", "Sunday"])

# Overview
popular_items = list(df['Item'].value_counts().head(5).index)
st.info(f"**Total Transactions**: {df['Transaction'].nunique()} \n\n **Most Popular Items**: {', '.join(popular_items)}")

# Get user inputs
def user_input():
    item_types = df['Item'].unique()
    period_types = df['period_day'].unique()
    
    item = st.selectbox('Item', item_types)
    period_day = st.selectbox('Period Day', period_types)
    weekday_weekend = st.selectbox('Weekday or Weekend', ['Weekday', 'Weekend'])
    month = st.select_slider('Month', 
                             ["January", "February", "March", "April", "May", "June", 
                              "July", "August", "September", "October", "November", 
                              "December"])
    day = st.select_slider('Day', 
                           ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", 
                            "Saturday", "Sunday"])
    
    return item, period_day, weekday_weekend, month, day

item, period_day, weekday_weekend, month, day = user_input()

# Get filtered data
def get_data(period_day, weekday_weekend, month, day):
    data = df.copy()
    
    filter = data.loc[
        (data['period_day'].str.contains(period_day)) & 
        (data['weekday_weekend'].str.contains(weekday_weekend)) & 
        (data['month'].str.contains(month)) & 
        (data['day'].str.contains(day))
    ]
    return filter if filter.shape[0] > 0 else None

data = get_data(period_day, weekday_weekend.lower(), month, day)

if data is None:
    st.warning(f"No transaction data found for the selected filters: {period_day}, {weekday_weekend}, {month}, {day}. Please try different filters.")
else:
    # One-hot encoding
    item_count = data.groupby(['Transaction', 'Item'])['Item'].count().reset_index(name='Count')
    item_count_pivot = item_count.pivot_table(index='Transaction', columns='Item', values='Count', aggfunc='sum').fillna(0)
    one_hot_df = item_count_pivot.map(lambda x: x > 0)  # Convert to boolean
    
    # Apriori algorithm
    threshold = 1
    support = 0.01 if len(data) > threshold else 0.005
    frequent_items = apriori(one_hot_df, min_support=support, use_colnames=True)
    
    # Generate association rules
    metric = 'lift'
    rules = association_rules(frequent_items, metric=metric, min_threshold=threshold)[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

    def parse_list(x):
        x = list(x)
        return x[0] if len(x) == 1 else ', '.join(x)

    def result(antecedents):
        data = rules[['antecedents', 'consequents']].copy()
        
        data['antecedents'] = data['antecedents'].apply(parse_list)
        data['consequents'] = data['consequents'].apply(parse_list)
        
        result = data.loc[data['antecedents'] == antecedents]
        return result.iloc[0] if not result.empty else None

    # Get the association result
    association_result = result(item)
    
    if association_result is not None:
        st.markdown('Result:')
        st.success(f"If customer purchases **{item}**, then they might also purchase **{association_result['consequents']}**.")
    else:
        st.info(f"No association rules found for the item **{item}**.")
