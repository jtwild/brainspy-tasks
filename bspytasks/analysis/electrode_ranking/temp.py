# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:56:18 2020

@author: Jochem
"""
tuple_filter = (slice(None), slice(None), 'darwin2', slice(None), 0)
capacity_temp =[]
nancounter_relative_temp = []
for ind in range(49):
    nancounter_relative_temp.append( df_vc.loc[tuple_filter, 'nancounter'].values[ind] / df_vc.loc[tuple_filter, 'found'].values[ind].size)
    capacity_temp.append( df_vc.loc[tuple_filter, 'found'].values[ind].sum() / df_vc.loc[tuple_filter, 'found'].values[ind].size)

df_vc.loc[tuple_filter, 'nancounter_relative'] = nancounter_relative_temp
df_vc.loc[tuple_filter, 'capacity'] = capacity_temp