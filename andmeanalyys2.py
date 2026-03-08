import xml.etree.ElementTree as ET
import pandas as pd
import json
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt
import re

# Loen baltic_psc_3m_metadata.xml failist välja out.json failis sisalduvad vead. Profileerin andmestiku. 

# Importin JSON faili https://www.geeksforgeeks.org/python/read-json-file-using-python/
with open('out.json', 'r') as file:
    data = json.load(file)

# Määran DataFrame analüüsiks mittetarvilike ridadeta
df = pd.DataFrame(data["deficiencies"])
print(df)

# Loon DataFarme (df_kirj), kuna tundub, et describe funktsioon võiks mind aidata andmete edasisel profileerimisel ning puhastamisel https://www.geeksforgeeks.org/pandas/python-pandas-dataframe-describe-method/
# Profileerimisel juhindun juhendist: https://www.geeksforgeeks.org/python/pandas-profiling-in-python/
df_kirj = df.describe(include='all')

# Missing values https://note.nkmk.me/en/python-pandas-nan-judge-count/#google_vignette
print("missing values:\n", df.isnull().sum())

# Unique values https://www.geeksforgeeks.org/python/python-pandas-series-nunique/
print("\nunique values:", df['inspection_id'].nunique())
print("unique values:", df['deficiency_code'].nunique())

# Data types https://www.geeksforgeeks.org/pandas/pandas-dataframe-dtypes/
print("\ndata types:")
print(df.dtypes)

# metadata näitas mulle kätte tulbad vigadega out.json andmestikus, hakkan järjest puhastama: 

# O-st nullid https://www.geeksforgeeks.org/data-analysis/python-pandas-dataframe-replace/                                                                     
df['deficiency_code'] = df['deficiency_code'].replace('O', '0')

# Eemaldan tühikud https://www.geeksforgeeks.org/pandas/pandas-strip-whitespace-from-entire-dataframe/
df['deficiency_description'] = df['deficiency_description'].str.strip()

# Kõik tähed suureks, et kuju oleks sama https://www.geeksforgeeks.org/pandas/python-pandas-series-str-lower-upper-and-title/
df['deficiency_description'] = df['deficiency_description'].str.upper()
df['deficiency_category'] = df['deficiency_category'].str.upper()

# Leidsin probleemi, et kood leidis kõige kallimad kulud, aga mitte keskmised. Seetõttu kasutan Regexit, et eemaldada tekstilised erisused. https://www.geeksforgeeks.org/pandas/replace-values-in-pandas-dataframe-using-regex/
# Terminal andis "float" errorit, ehk leidsin rohu https://www.statology.org/typeerror-expected-string-or-bytes-like-object/
"""
def clean_kirj(name):
    return re.sub(r"\[.*\]", "", name).strip()
"""

def clean_kirj(name):
    text = str(name) if pd.notnull(name) else ""
    return re.sub(r"\[.*\]", "", text).strip()

df['deficiency_description'] = df['deficiency_description'].apply(clean_kirj)
print(df)

# Duplikaadid minema, kust vaja https://www.geeksforgeeks.org/pandas/how-to-drop-rows-with-nan-values-in-pandas-dataframe/
df = df.dropna(subset=['inspection_id', 'deficiency_code'])
df = df.drop_duplicates(subset=['inspection_id', 'deficiency_code'])

# Maksumus ühel kujul, teised NaN ja hind suurem kui 0 https://www.geeksforgeeks.org/python/python-pandas-to_numeric-method/
# Regex aitaks ka siin, sest väärtustest on sees tähti, kuid on juba hilja - https://regex101.com
df['est_rectification_cost_eur'] = pd.to_numeric(df['est_rectification_cost_eur'], errors='coerce')
df = df[df['est_rectification_cost_eur'] > 0]

# Määratud ajavahemik https://www.geeksforgeeks.org/python/how-to-filter-dataframe-rows-based-on-the-date-in-pandas/
df['inspection_date'] = pd.to_datetime(df['inspection_date'], errors='coerce')
start_date = '2015-01-01'
end_date = '2026-03-31'
df = df[df['inspection_date'].between(start_date, end_date)]

# Loon Apriori jaoks andmetest ühe "ostukorvi" (concatenate) https://www.geeksforgeeks.org/python/concatenate-strings-from-several-rows-using-pandas-groupby/
transactions = df.groupby('inspection_id')['deficiency_description'].apply(list).values.tolist()

# Implementeerin Apriori https://www.geeksforgeeks.org/machine-learning/implementing-apriori-algorithm-in-python/
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_array, columns=te.columns_)

# Tagantjärele - terminalis näitas vaid AIS süsteemiga seotud assotsatsiooni, pean selle seisu üldisemaks saama
# Proovin supporti madaldada ning, et AIS-ist mööda vaadata, leida mooduse lifti järgi sorteerimiseks https://www.geeksforgeeks.org/python/sorting-rows-in-pandas-dataframe/
frequent_itemsets = apriori(df_encoded, min_support=0.005, use_colnames=True)
print("Total Frequent Itemsets:", frequent_itemsets.shape[0])

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
rules = rules[rules['antecedents'].apply(lambda x: len(x) >= 1) & rules['consequents'].apply(lambda x: len(x) >= 1)]
rules_lift = rules.sort_values(by='lift', ascending=False)
# Tagantjärele 2 - sain lift järgi sorteeritud, aga A+B pole piisavalt pikal kujul Terminalis, seega https://stackoverflow.com/questions/11707586/how-do-i-expand-the-output-display-to-see-more-columns-of-a-pandas-dataframe
pd.set_option('display.max_columns', None)  
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 2000)
pd.set_option('display.expand_frame_repr', False)

print("Association Rules:", rules.shape[0])
print(rules_lift[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

# Top 10 esinemine https://www.geeksforgeeks.org/pandas/python-pandas-series-value_counts/
top_10_vead = df['deficiency_description'].value_counts().head(10)
print(top_10_vead)

# Top 10 maksumus https://www.geeksforgeeks.org/pandas-groupby-summarizing-data/
kulu = df.groupby('deficiency_description')['est_rectification_cost_eur'].mean().reset_index()
top_10_kulu = kulu.sort_values(by='est_rectification_cost_eur', ascending=False).head(10)
print(top_10_kulu)

# https://www.geeksforgeeks.org/pandas/pandas-groupby/
tabel = df.groupby('deficiency_description').agg({
    'deficiency_description': 'count', 
    'est_rectification_cost_eur': 'mean'
})
# Riskiskoor - allika kaotasin ära
tabel['riskiskoor'] = tabel['deficiency_description'] * tabel['est_rectification_cost_eur']
tabel = tabel.sort_values(by='riskiskoor', ascending=False)
print(tabel.head(10))

# KOKKUVÕTTEKS: Palju ringe ümber pudru ja kõht ikka tühi. Eeltöötlus jäi poolikuks lõpuks ikkagi, GIGO. 