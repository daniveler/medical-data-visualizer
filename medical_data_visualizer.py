# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#%%
# 1
df = pd.read_csv('medical_examination.csv')

df.head()

#%%
# 2
df['overweight'] = np.where(df['weight'] / ((df['height'] / 100) ** 2) > 25, 1, 0)

df.head()
#%%
# 3
df['cholesterol'] = np.where(df['cholesterol'] > 1, 1, 0)
df['gluc'] = np.where(df['gluc'] > 1, 1, 0)
df.head(10)

# %%
# 4
def draw_cat_plot():
    # %%
    # 5
    df_cat = pd.melt(
        df, 
        id_vars= ['id'], 
        value_vars= ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'], 
        var_name='variable', 
        value_name= 'value')
    


   # %% 
    # 6
    df_cat = pd.melt(
        df, 
        id_vars= ['cardio'], 
        value_vars= ['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'], 
        var_name='variable', 
        value_name= 'value')
    
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name = 'total')

    df_cat.head(10000)


   # %% 
    # 7
    

    # %%
    # 8
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', kind='bar', data=df_cat)

    # %%
    # 9
    fig.savefig('catplot.png')
    return fig

# %%
# 10
def draw_heat_map():
    # %%
    # 11
    df_heat = df.loc[
        (df['ap_lo'] <= df['ap_hi']) & 
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975)) 
    ]

    df_heat.head()

    # %%
    # 12
    corr = df_heat.corr()

    corr

    # %%
    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # %%
    # 14
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))

    # 15
    sns.heatmap(data=corr, mask=mask, annot=True, cmap='viridis', fmt=".1f")

    # %%
    # 16
    fig.savefig('heatmap.png')
    return fig
