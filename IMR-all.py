#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from scipy import stats
from matplotlib import pyplot as plt
import numpy as np
from statsmodels.stats import multicomp
import requests
import os
import scipy.stats as stats
import gmaps
from config import gkey
from scipy.stats import linregress

# Configure gmaps
gmaps.configure(api_key=gkey)


# ## Infant Mortaility by Race

# In[2]:


dfImrByCountyByRace = pd.read_csv("datafiles/IMR by county by race, 2007-2016.txt", sep='\t')
# dfImrByCountyByRace['Race'] = ["Unknown" if race is np.nan else race for race in dfImrByCountyByRace['Race']]

dfImrByCountyByRace.dropna(subset=['Death Rate'], inplace=True)


# In[3]:


# remove rows with unreliable death rate data (fewer than 20 deaths)
dfImrByCountyByRace = dfImrByCountyByRace.loc[dfImrByCountyByRace['Death Rate'].map(lambda x: 'Unreliable' not in str(x))]
# dfImrByCountyByRace.loc[dfImrByCountyByRace['Race'] is np.nan]

# convert death rate to float
dfImrByCountyByRace['Death Rate'] = dfImrByCountyByRace['Death Rate'].map(lambda x: float(x))


# In[4]:


blacks = dfImrByCountyByRace.loc[dfImrByCountyByRace['Race Code'] == '2054-5']['Death Rate']
natives = dfImrByCountyByRace.loc[dfImrByCountyByRace['Race Code'] == '1002-5']['Death Rate']
whites = dfImrByCountyByRace.loc[dfImrByCountyByRace['Race Code'] == '2106-3']['Death Rate']
asians = dfImrByCountyByRace.loc[dfImrByCountyByRace['Race Code'] == 'A-PI']['Death Rate']
unknown = dfImrByCountyByRace.loc[dfImrByCountyByRace['Race'] == 'Unknown']['Death Rate']


# In[5]:


ax = dfImrByCountyByRace.boxplot('Death Rate', by="Race", figsize=(10, 6))
fig = ax.get_figure()
ax.set_xticklabels (['American Indian/\nAlaska native', 'Asian/\nPacific Islander', 'Black/\nAfrican American', 'White'])
ax.set_ylabel('Death Rate (per 1000 births)')
fig.savefig('Images/Death Rate by Race.png')
fig.show()


# ### ANOVA shows that one (or more) race(s) is significantly different than the rest

# In[6]:


stats.f_oneway(blacks, natives, whites, asians)


# In[7]:


blacksByCounty = dfImrByCountyByRace.set_index(['Race', 'County']).sort_values(['Race', 'Death Rate'], ascending=False).loc['Black or African American']
blacksByCounty = blacksByCounty.loc[blacksByCounty['Death Rate'].notnull()]


# In[8]:


blacksHighestImrCounties = blacksByCounty.head(10)
blacksHighestImrCounties.to_csv("datafiles/AfricanAmericanHighestImrCounties.csv")


# In[9]:


blacksLowestImrCounties = blacksByCounty.tail(10)
blacksLowestImrCounties.to_csv("datafiles/AfricanAmericanLowestImrCounties.csv")


# In[10]:


# use pairwise tukeyhsd to find out which race is significnalty different than the rest  
answer = multicomp.pairwise_tukeyhsd(dfImrByCountyByRace['Death Rate'], dfImrByCountyByRace['Race'], alpha=0.05)


# In[11]:


# reject True proves the hypothesis - that there is significant difference between two means
print(answer)


# ## Infant Mortality by Race, 2007-2016

# In[12]:


dfImrByYearByRace = pd.read_csv("datafiles/imr by year by race, 2007-2016.txt", sep='\t')


# In[13]:


dfImrByYearByRace.dropna(subset=['Year of Death'], inplace=True)
dfImrByYearByRace['Race'] = ['Unknown' if myrace is np.nan else myrace for myrace in dfImrByYearByRace['Race']]


# In[14]:


dfImrByYearByRace.head()


# In[15]:


dfPlot = dfImrByYearByRace.pivot('Year of Death', 'Race', 'Death Rate')
dfPlot.reset_index(inplace=True)


# In[16]:


dfPlot.head()


# In[17]:


plt.plot(dfPlot['Year of Death'], dfPlot['Unknown'], label='Unknown')
plt.plot(dfPlot['Year of Death'], dfPlot['American Indian or Alaska Native'], label='American Indian or Alaska Native')
plt.plot(dfPlot['Year of Death'], dfPlot['Asian or Pacific Islander'], label='Asian or Pacific Islander')
plt.plot(dfPlot['Year of Death'], dfPlot['Black or African American'], label='Black or African American')
plt.plot(dfPlot['Year of Death'], dfPlot['White'], label='White')
plt.ylim(0, 20)
plt.legend()
plt.xlabel("Year")
plt.ylabel("Death Rate (per 1000)")
plt.title ("IMR by Race, 2007-2016")
plt.rcParams["figure.figsize"] = [8, 6]
plt.savefig("Images/IMR by Race, 2007-2016.png")
plt.show()


# ## Leading cause of infant mortality by age
# 

# In[18]:


# Death Rate by age
dfImrByAgeByCause = pd.read_csv("datafiles/IMR by age by cause, 2007-2016.txt", sep='\t')

# Convert NaN to Unknown for the age of infant
dfImrByAgeByCause['Age of Infant at Death'] = ["Unknown" if age is np.nan else age for age in dfImrByAgeByCause['Age of Infant at Death']]

# Remove rows with Unreliable in Death Rate column (fewer than 20 reported cases)
dfImrByAgeByCause = dfImrByAgeByCause.loc[dfImrByAgeByCause['Death Rate'].map(lambda x: 'Unreliable' not in str(x))]

# Convert Death Rate to float
dfImrByAgeByCause['Death Rate'] = dfImrByAgeByCause['Death Rate'].map(lambda x: float(x))

# Save totals in new df
dfTotalsbyAgebyCause = dfImrByAgeByCause.loc[(dfImrByAgeByCause['Notes']=='Total') & (dfImrByAgeByCause['Age of Infant at Death']!= 'Unknown')]

# Remove Totals and summary rows
dfImrByAgeByCause = dfImrByAgeByCause[(dfImrByAgeByCause['Notes']!="Total")& (dfImrByAgeByCause["Death Rate"].notnull())]

# Sort descending by Death Rate
dfImrByAgeByCause.sort_values(by='Death Rate', ascending=False, inplace=True)

dfImrByAgeByCause.head()


# In[19]:


rects = plt.barh(dfTotalsbyAgebyCause['Age of Infant at Death'], width=dfTotalsbyAgebyCause['Death Rate'], color='blue', alpha=.5, edgecolor='black')
plt.title("IMR vs Age")
plt.ylabel("Age of Infant at Death")
plt.xlabel("Death Rate")
plt.xlim(0, 2.5)
plt.rcParams["figure.figsize"] = [10, 6]
plt.tight_layout()
plt.savefig("Images/IMR vs age.png")
plt.show()


# In[20]:


# removes rows that show total
dfImrByAgeByCause = dfImrByAgeByCause.loc[(dfImrByAgeByCause['Notes'] != 'Total')].sort_values(by=['Death Rate'], ascending=False).head(20)


# In[21]:


indexedImrByAgeByCause = dfImrByAgeByCause.set_index(["Age of Infant at Death", "Cause of death"])
indexedImrByAgeByCause.sort_values(["Age of Infant at Death Code", "Death Rate"], ascending=[True, False], inplace=True)
indexedImrByAgeByCause.head()


# In[22]:


x_axis = dfImrByAgeByCause['Cause of death'].head(6)

y_axis = dfImrByAgeByCause['Death Rate'].head(6)
rects = plt.bar(range(len(x_axis)),y_axis, color='teal', alpha=0.5, edgecolor='black' )

plt.xticks(range(len(x_axis)), ['Extreme\nImmaturity', 'Sudden\nInfant Death\nSyndrom\nSIDS', 'Extreme\nImmaturity', 'Other\nIll-defined/\nUnspecificed\nCauses', 'Accidental\nSuffocation\nStrangulation\nin bed', 'Other\nPreterm\nInfants'], rotation='horizontal')
plt.xlim(-1,6)
plt.ylim(0, 0.65)
for rect in rects:
    indx = rects.index(rect)
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width()/1.1, height + 0.02,
             dfImrByAgeByCause['Age of Infant at Death'].iloc[indx],
             ha='center', va='bottom', color='purple', rotation = 45)
plt.text(4.5,0.6, "Infant's Age at Death", color="purple", bbox=dict(facecolor='none', edgecolor='purple'))
   
plt.xlabel("Cause of Death")
plt.ylabel("Mortality Rate (per 1000)")
plt.title("Leading Causes of Infant Mortality")
plt.rcParams["figure.figsize"] = [12, 8]
# plt.tight_layout()
plt.savefig("Images/Leading Causes of Infant Mortality.png")
plt.show()


# In[23]:


# dfImrByAgeByCause.plot('Cause of death', "Death Rate", kind='bar')


# ## Leading cause of infant mortality by race
# 

# In[24]:


dfImrByRaceByCause = pd.read_csv("datafiles/imr by race by cause, 2007-2016.txt", sep='\t')

# remove Unreliable from Death Rate colum (fewer than 20 reported cases)
dfImrByRaceByCause = dfImrByRaceByCause.loc[dfImrByRaceByCause['Death Rate'].map(lambda x: 'Unreliable' not in str(x))]

# remove totals
dfImrByRaceByCause = dfImrByRaceByCause[(dfImrByRaceByCause['Notes']!='Total') & dfImrByRaceByCause['Death Rate'].notnull()]

# convert death rate to float
dfImrByRaceByCause['Death Rate'] = dfImrByRaceByCause['Death Rate'].map(lambda x: float(x))

# sort by race ascending, death rate descending
dfImrByRaceByCause = dfImrByRaceByCause.sort_values(['Race', 'Death Rate'], ascending=[True, False])

dfImrByRaceByCause.head()


# In[25]:


causes = x_axis.unique()
races = dfImrByRaceByCause['Race'].unique()


# In[26]:


mylist = []
for cause in causes:
    for thisrace in races:
        if ((dfImrByRaceByCause['Cause of death'] == cause) & (dfImrByRaceByCause['Race'] == thisrace)).any():
            myrow = dfImrByRaceByCause.loc[(dfImrByRaceByCause['Cause of death'] == cause) & (dfImrByRaceByCause['Race'] == thisrace)]
            mylist.append({
                'Race': thisrace,
                'Cause of death': cause,
                'Death Rate' : myrow.iloc[0]['Death Rate']})
        else:
            mylist.append({
                'Race': thisrace,
                'Cause of death': cause,
                'Death Rate' : 0})

dfImrByRaceByCausePlotting = pd.DataFrame(mylist)

dfImrByRaceByCausePlotting = dfImrByRaceByCausePlotting.pivot(index='Race', columns='Cause of death', values='Death Rate')

dfImrByRaceByCausePlotting = dfImrByRaceByCausePlotting[[
 'Extreme immaturity',
 'Sudden infant death syndrome - SIDS',
 'Other ill-defined and unspecified causes of mortality',
 'Accidental suffocation and strangulation in bed',
 'Other preterm infants']]


# In[27]:


ax = dfImrByRaceByCausePlotting.plot.bar(stacked=True, ylim=(0,7), figsize=(11, 9),  width=0.75, title="Leading Causes of IMR by Race", rot=0)
ax.set_ylabel("Death Rate (per 1000)")
ax.set_xticklabels (['American Indian/\nAlaska native', 'Asian/\nPacific Islander', 'Black/\nAfrican American', 'White'])
fig = ax.get_figure()
fig.savefig('Images/Leading Causes of IMR by Race.png')


# ## IMR by Month Prenatal Care Began
# 

# In[28]:


dfImrByPrenatalCareStart = pd.read_csv("./datafiles/imr by prenatal care start, 2007-2016.txt", sep='\t')


# In[29]:


dfImrByPrenatalCareStart = dfImrByPrenatalCareStart[['Month Prenatal Care Began', 'Death Rate']].dropna(subset=["Month Prenatal Care Began", "Death Rate"])


# In[30]:


dfImrByPrenatalCareStart


# In[31]:


dfImrByPrenatalCareStart.plot("Month Prenatal Care Began", "Death Rate", kind='bar')


# In[32]:


dfImrByPrenatalByRace = pd.read_csv("datafiles/IMR By prenatal care by race, 2007-2016.txt", sep='\t')


# In[33]:


dfNoPrenatalcareByRace = dfImrByPrenatalByRace.loc[(dfImrByPrenatalByRace['Month Prenatal Care Began'] == 'No prenatal care') & (dfImrByPrenatalByRace['Notes']!='Total')]
dfNoPrenatalcareByRace['Death Rate'] = [float(x) for x in dfNoPrenatalcareByRace['Death Rate']]


# In[34]:


df1stMonthPrenatalcareByRace = dfImrByPrenatalByRace.loc[(dfImrByPrenatalByRace['Month Prenatal Care Began'] == '1st month') & (dfImrByPrenatalByRace['Notes']!='Total')]
df1stMonthPrenatalcareByRace['Death Rate'] = [0 if 'Unreliable' in str(x) else x for x in df1stMonthPrenatalcareByRace['Death Rate']]
df1stMonthPrenatalcareByRace['Death Rate'] = [float(x) for x in df1stMonthPrenatalcareByRace['Death Rate']]


# In[35]:


# df1stMonthPrenatalcareByRace


# In[36]:


plt.bar(np.arange(len(dfNoPrenatalcareByRace['Race'])), dfNoPrenatalcareByRace['Death Rate'], label='No Prenatal Care', width=0.25)
plt.bar(np.arange(len(df1stMonthPrenatalcareByRace['Race'])) + 0.25, df1stMonthPrenatalcareByRace['Death Rate'], label='1st Month Prenatal Care', width=0.25)
plt.xticks(np.arange(len(dfNoPrenatalcareByRace['Race'])), dfNoPrenatalcareByRace['Race'])
plt.legend()


# ## Percent of Premature Births by Race

# In[37]:


dfExtremePrematurityByRace = pd.read_csv("datafiles/extreme prematurity by race, 2007-2017.txt", sep='\t')


# In[38]:


# dfExtremePrematurityByRace.dropna(subset=['Births'])


# In[39]:


def totalForRace(df, race):
    total = df.loc[df['Bridged Race'] == race]['Births'].mean()
    return total
    


# In[40]:


dfExtremePrematurityByRace.dropna(subset=['Births'], inplace=True)


# In[41]:


totalForRace(dfExtremePrematurityByRace, "White")


# In[42]:


totalForRace(dfExtremePrematurityByRace, "Black or African American")


# In[43]:


dfExtremePrematurityByRace['Total for Race'] = dfExtremePrematurityByRace.apply((lambda x: totalForRace(dfExtremePrematurityByRace, x['Bridged Race'])), axis=1) 


# In[44]:


# dfExtremePrematurityByRace


# In[45]:


dfExtremePrematurityAfricanAmerican = dfExtremePrematurityByRace.loc[dfExtremePrematurityByRace['Bridged Race'] == 'Black or African American']


# In[46]:


# dfExtremePrematurityAfricanAmerican


# In[47]:


dfExtremePrematurity20to27 = dfExtremePrematurityByRace.loc[dfExtremePrematurityByRace['OE Gestational Age 10'] == '20 - 27 weeks']


# In[48]:


dfExtremePrematurity20to27['Percent of Total Births'] = dfExtremePrematurity20to27.apply((lambda row: row['Births']/row['Total for Race']*100), axis=1)


# In[49]:


dfExtremePrematurity20to27


# ## US IMR Compared to Other Wealthy Nations

# In[50]:


dfWhoByCountry = pd.read_csv("datafiles/who by country.csv.csv", header=1)


# In[51]:


dfWhoByCountry.rename(columns={'Both sexes':'IMR'}, inplace=True)
dfWhoByCountry.loc[dfWhoByCountry['Country'] == 'United Kingdom of Great Britain and Northern Ireland', 'Country'] = 'United Kingdom'
dfWhoByCountry.head()


# In[52]:


plotImrByCountry = dfWhoByCountry.groupby('Country').mean().sort_values(by='IMR', ascending=False)


# In[53]:


plotImrByCountry.head()


# In[54]:


rects = plt.barh(plotImrByCountry.index, plotImrByCountry['IMR'], alpha=0.5, edgecolor='black')
plt.title("IMR 2007-2017 ")
plt.xlabel("Infant Deaths per 1000 Births")

for rect in rects:
    indx = rects.index(rect)
    width = rect.get_width()

    plt.text(width - 0.2, rect.get_y()+0.2,
         plotImrByCountry['IMR'].iloc[indx].round(2),
         ha='center', va='bottom', color='black', rotation = 'horizontal')
    
rects[0].set_color('red')
plt.rcParams["figure.figsize"] = [18, 10]
plt.savefig("Images/imr_by_country.png")
plt.tight_layout()
plt.show()


# In[ ]:





# ### Education levels vs Death Rate in USA by CDC

# In[55]:


filename = 'datafiles/Education_Infant_Death_Records_2007_2016.csv'
filename_df = pd.read_csv(filename, encoding="ISO-8859-1")


# In[56]:


education_sorted =filename_df.sort_values(["Death Rate"],ascending=False)


# In[57]:


exclude_unknown = education_sorted.loc[education_sorted['Education']!= "Unknown/Not on certificate"]


# In[58]:


del exclude_unknown['Deaths']
del exclude_unknown['Education Code']
del exclude_unknown['Births']
del exclude_unknown['Notes']
exclude_unknown


# In[ ]:





# In[59]:


x_axis = exclude_unknown['Education']
y_axis = exclude_unknown['Death Rate']

# plt.plot(exclude_unknown["Education"],
#          exclude_unknown["Death Rate"]
         
#          )
plt.plot(x_axis, y_axis)


# Incorporate the other graph properties
plt.style.use('seaborn')
plt.title(f"Death rate by Education level 2007-2016")
plt.ylabel("Death Rate (per 1000)")
plt.xlabel("Education level")
plt.grid(True)
plt.xticks((range(len(x_axis))), ["9-12th\ngrade", "HS\nor GED", "Excluded", "8th\ngrade\nor less", "Some\ncollege\ncredit", "Associate\ndegree",  "Bachelor's\ndegree", "Master's\ndegree", "Doctorate\ndegree"])
plt.xlim(-0.5, 8.2)
plt.ylim(3, 9)
# Save the figure
plt.savefig("Images/Education_lever_line.png")

# Show plot
plt.show()


# In[60]:


x_axis = exclude_unknown['Education']
y_axis = exclude_unknown['Death Rate']
plt.tight_layout()
plt.ylabel("Death Rate per 1000")
plt.xlabel("Education level 2007-2016")
plt.bar(x_axis, y_axis, color="b", align="center")
plt.xticks(exclude_unknown['Education'], rotation="vertical")

plt.savefig("Images/Education_level_barchart.png")
plt.show()


# ### Education level in USA by USDA

# In[61]:


usdafilename = 'datafiles/Education_USDA.csv'
usdafilename_df = pd.read_csv(usdafilename, encoding="ISO-8859-1")


# In[62]:


del usdafilename_df['Less than a high school diploma, 2013-17']
del usdafilename_df['High school diploma only, 2013-17']
del usdafilename_df["Some college or associate's degree, 2013-17"]
del usdafilename_df["Bachelor's degree or higher, 2013-17"]
del usdafilename_df["Unnamed: 11"]
        


# In[63]:


usdafilename_df = usdafilename_df.dropna()
         


# In[64]:


renamed_code = usdafilename_df.rename(columns={"FIPS Code":"GEOID"})
renamed_code['GEOID'] = renamed_code['GEOID'].map(lambda x: int(x))
renamed_code


# In[65]:


gazfilename = 'datafiles/2017_Gaz_counties_national.csv'
gazfilename_df = pd.read_csv(gazfilename, encoding="ISO-8859-1")


# In[66]:


del gazfilename_df['USPS']
del gazfilename_df['ANSICODE']
del gazfilename_df["NAME"]
del gazfilename_df["ALAND"]
del gazfilename_df['AWATER']
del gazfilename_df['ALAND_SQMI']
del gazfilename_df["AWATER_SQMI"]
gazfilename_df.head()


# In[67]:


merge_table = pd.merge(renamed_code, gazfilename_df, on="GEOID")


# In[68]:


merge_table.columns


# In[69]:


merge_table.columns = ['GEOID', 'State', 'Area name',
       'Percent of adults with less than a high school diploma, 2013-17',
       'Percent of adults with a high school diploma only, 2013-17',
       'Percent of adults completing some college or associate\'s degree, 2013-17',
       'Percent of adults with a bachelor\'s degree or higher, 2013-17',
       'INTPTLAT',
       'INTPTLONG']
merge_table.head()


# In[70]:


def weighted_education(row):
    a = row['Percent of adults with less than a high school diploma, 2013-17']*0
    b = row['Percent of adults with a high school diploma only, 2013-17']*0.4
    c = row["Percent of adults completing some college or associate's degree, 2013-17"]*0.6
    d = row["Percent of adults with a bachelor's degree or higher, 2013-17"]*0.8
    return (a+b+c+d)


# In[71]:


merge_table['Weighted Education Score'] = merge_table.apply(weighted_education, axis=1)


# In[72]:


merge_table


# In[73]:


# Store latitude and longitude in locations
locations = merge_table[["INTPTLAT", "INTPTLONG"]]
# Plot Heatmap
fig = gmaps.figure()
BD = merge_table["Weighted Education Score"]
# Create heat layer
heat_layer = gmaps.heatmap_layer(locations, weights=BD , 
                                 dissipating=False, max_intensity=100,
                                 point_radius=0.5)


# Add layer
fig.add_layer(heat_layer)

# Display figure
fig


# ### Education vs High IMR by counties

# In[74]:


highcounties = 'datafiles/high_IMR_county.csv'
highcounties_df = pd.read_csv(highcounties, encoding="ISO-8859-1")
highcounties_df


# In[75]:


coordinates3 = [
    (32.577195, -93.882423),
    (30.543930, -91.093131),
    (32.267788, -90.466017),
    (39.300032, -76.610476),
    (33.553444, -86.896536),
    (35.183794, -89.895397),
    (38.635699, -90.244582),
    (39.196927, -84.544187),
    (42.284664, -83.261953),
    (35.050192, -78.828719)
]


# In[76]:


figure_layout3 = {
    'width': '400px',
    'height': '300px',
    'border': '1px solid black',
    'padding': '1px',
    'margin': '0 auto 0 auto'
}
fig3 = gmaps.figure(layout=figure_layout3)


# In[77]:


# Assign the marker layer to a variable
markers3 = gmaps.marker_layer(coordinates3)
# Add the layer to the map
fig.add_layer(markers3)
#fig


# In[78]:


fig = gmaps.figure()

fig.add_layer(heat_layer)
fig.add_layer(markers3)

fig


# ### Education vs Low IMR by counties

# In[79]:


lowcounties = 'datafiles/low_IMR_county.csv'
lowcounties_df = pd.read_csv(lowcounties, encoding="ISO-8859-1")
lowcounties_df


# In[80]:


coordinates4 = [
    (38.051817, -122.745974),
    (39.865669, -74.258864),
    (37.414672, -122.371546),
    (39.325414, -104.925987),
    (40.959698, -74.074727),
    (40.858896, -74.547292),
    (37.220777, -121.690622),
    (37.727239, -123.032229),
    (40.565527, -74.619938),
    (40.287048, -74.152446)
]


# In[81]:


figure_layout4 = {
    'width': '400px',
    'height': '300px',
    'border': '1px solid black',
    'padding': '1px',
    'margin': '0 auto 0 auto'
}
fig4 = gmaps.figure(layout=figure_layout4)


# In[82]:


# Assign the marker layer to a variable
markers4 = gmaps.marker_layer(coordinates4)
# Add the layer to the map
fig.add_layer(markers4)
#fig


# In[83]:


fig = gmaps.figure()

fig.add_layer(heat_layer)
fig.add_layer(markers4)

fig


# ### AAR Education vs Death Rate

# In[84]:


aareducation = 'datafiles/Death_Rate_by_AAR_by_Education.csv'
aareducation_df = pd.read_csv(aareducation, encoding="ISO-8859-1")


# In[85]:


aareducation_sorted =aareducation_df.sort_values(["Death Rate"],ascending=False)


# In[86]:


exclude_unknown_aar = aareducation_sorted.loc[aareducation_sorted['Education']!= "Unknown/Not on certificate"]
exclude_unknown_aar


# In[87]:


x_axis = exclude_unknown_aar['Education']
y_axis = exclude_unknown_aar['Death Rate']
plt.tight_layout()
plt.ylabel("Death Rate %")
plt.xlabel("Education level of African American Race")
plt.bar(x_axis, y_axis, color="b", align="center")
plt.xticks(range(len(x_axis)), ["9-12th\ngrade", "Excluded", "HS\nor GED", "Some\ncollege\ncredit" , "Associate\ndegree", "8th\ngrade\nor less", "Bachelor's\ndegree", "Master's\ndegree", "Doctorate\ndegree"] )

plt.savefig("Images/Education_of_AAR.png")
plt.show()


# In[88]:


censusaareducation = 'datafiles/AAR_Education_2013_2017.csv'
censusaareducation_df = pd.read_csv(censusaareducation, encoding="ISO-8859-1")


# In[89]:


censusaareducation_df.dropna()


# ### Level of Education of African Americans 2013-2017

# In[90]:


x_axis = censusaareducation_df['Education']
y_axis = censusaareducation_df['Average']
plt.tight_layout()
plt.ylabel("Average %")
plt.xlabel("Education level of African American Race")
plt.bar(x_axis, y_axis, color="b", align="center")
plt.xticks(censusaareducation_df['Education'], rotation="vertical")

plt.savefig("Images/Census_Education_of_AAR.png")
plt.show()


# In[ ]:





# In[91]:


file= "datafiles/birth_weight.csv"
birthweight_df=pd.read_csv(file)
birthweight_df


# In[92]:


total_rate = list(birthweight_df["Death Rate Per 1,000"])


# In[93]:


birthweight_df


# In[94]:


list_rate = []
sum_rate = birthweight_df["Death Rate Per 1,000"].sum()
for rate in total_rate:
    rate_percent = (rate / sum_rate) * 100
    list_rate.append(rate_percent)


# In[95]:


birthweight_df["Deaths Rate"] =list_rate
birthweight_df["Deaths Percent (%)"] = birthweight_df["Deaths Rate"].map("{0:.2f}%".format)


# In[96]:


birthweight_df


# In[97]:


rate_percent = list(birthweight_df["Deaths Rate"])
birth_weight = list(birthweight_df["Birth Weight Code"])


# In[98]:


bw_x_axis = birth_weight
bw_y_axis = rate_percent


# In[99]:


birthweight_df["Birth Weight Code"]


# In[100]:


birthweight_df["Deaths Rate"]


# In[101]:


label = [" ~ 0.5kg", "~0.9kg", "~1.5kg", "~ 1.9kg", "~2.5kg", "~3.0kg", "~3.5kg","~4.0kg","~4.5kg", "~5.0kg","5.0~8.1 kg","Not Stated"]
plt.figure(figsize=(10,8))
plt.bar(bw_x_axis, bw_y_axis, color = "#D35400", alpha = 0.5,align ="center")
plt.title(f"Birth Weights Infant Mortality Rate",fontsize=15)
plt.xlabel("Weights", fontsize= 10)
plt.ylabel("Infant Mortality Rate",fontsize= 10)
plt.xticks(bw_x_axis, label, fontsize=10, rotation=30)
plt.savefig("Images/bw_Birth Weights Infant Mortality Rate.png")
plt.show()


# In[102]:


file= "datafiles/total_low_brith_weight_by_race.csv"
total_birthweight_df=pd.read_csv(file)
total_birthweight_df


# In[103]:


for i in range(6,10):
    total_birthweight_df[f"200{i}"]=total_birthweight_df[f"200{i}"].str.replace("%", "")


# In[104]:


for i in range(10,16):
    total_birthweight_df[f"20{i}"]=total_birthweight_df[f"20{i}"].str.replace("%", "")


# In[105]:


total_birthweight_df


# In[106]:


for i in range(6,10):
    total_birthweight_df[f"200{i}"]=total_birthweight_df[f"200{i}"].str.replace(",", "")
    total_birthweight_df[f"200{i}"]=total_birthweight_df[f"200{i}"].astype(float)


# In[107]:


for i in range(10,16):
    total_birthweight_df[f"20{i}"]=total_birthweight_df[f"20{i}"].str.replace(",", "")
    total_birthweight_df[f"20{i}"]=total_birthweight_df[f"20{i}"].astype(float)


# In[108]:


total_birthweight_df['2006'].astype(float)
total_birthweight_df.dtypes


# In[109]:


percent_race_df = pd.DataFrame(total_birthweight_df[total_birthweight_df["Data Type"].isin(["Percent"])])
total_race_df =pd.DataFrame(total_birthweight_df[total_birthweight_df["Data Type"].isin(["Number"])])


# In[110]:


race_list = list(total_race_df["Race"])


# In[111]:


race_list = race_list[0:5]


# In[112]:


percent_race_df=percent_race_df.iloc[0:5]


# In[113]:


percent_race_df["Race"] = race_list


# In[114]:


percent_race_df=percent_race_df.set_index("Race")
total_race_df=total_race_df.set_index("Race")


# In[115]:


percent_race_df = percent_race_df.drop("Data Type",axis=1)


# In[116]:


total_race_df=total_race_df.drop("Data Type",axis=1)


# In[117]:


average_percent_race_list = []
for race in race_list :
    average_percent_race_list.append(percent_race_df.loc[f"{race}"].mean())


# In[118]:


percent_race_df["Average %"] = average_percent_race_list


# In[119]:


percent_race_df.dtypes


# In[120]:


percent_race_df


# In[121]:


bw_race_x_axis = race_list 
bw_race_y_axis = list(percent_race_df["Average %"])


# In[122]:


bw_race_y_axis


# In[123]:


label = ['American Indian', 'Asian and Pacific Islander', 'African American', 'Hispanic or Latino', 'Non-Hispanic White']
plt.figure(figsize=(7,5))
plt.bar(bw_race_x_axis, bw_race_y_axis, color = "#7FB3D5", alpha = 0.5,align ="center")
plt.title(f"Birth Weights Infant Mortality Rate by Race",fontsize=15)
plt.xlabel("Race", fontsize= 10)
plt.ylabel("Infant Mortality Rate (Average %)",fontsize= 10)
plt.xticks(bw_race_x_axis, label, fontsize=10, rotation=90)
plt.savefig("Images/bw_Birth Weights Infant Mortality Rate by Race.png")
plt.show()


# In[124]:


file= "datafiles/overweight_rates.csv"
overweight_df=pd.read_csv(file)
overweight_df


# In[125]:


overweight_df = overweight_df.set_index("Location")


# In[126]:


list_overweight = list(overweight_df.loc['United States'])


# In[127]:


list_overweight


# In[128]:


overweight_df.dtypes


# In[129]:


label = ['American Indian', 'Asian and Pacific Islander', 'African American', 'Hispanic or Latino', 'Non-Hispanic White']
plt.figure(figsize=(10,8))
plt.bar(label,list_overweight, color = "#8E44AD", alpha = 0.5,align ="center")
plt.title(f"Overweight and Obesity Rates for Adults by Race/Ethnicity",fontsize=15)
plt.xlabel("Race", fontsize= 10)
plt.ylabel("Average %",fontsize= 10)
plt.xticks(fontsize=10, rotation=90)
plt.ylim(0,100)
plt.savefig("Images/bw_Overweight and Obesity Rates for Adults by Race.png")
plt.show()
#plt.xticks(x_axis, label, fontsize=10, rotation=90)


# In[130]:


file= "datafiles/total_rate_hypertension_race.csv"
hypertension_df=pd.read_csv(file)
hypertension_df


# In[131]:


hypertension_df.dtypes


# In[132]:


hypertension_df =hypertension_df.set_index("Location")


# In[133]:


list_hypertension=list(hypertension_df.loc["United States"])


# In[134]:


list_hypertension


# In[135]:


label = ['American Indian', 'Asian and Pacific Islander', 'African American', 'Hispanic or Latino', 'Non-Hispanic White']
plt.figure(figsize=(10,8))
plt.bar(label,list_hypertension, color = "#EC7063", alpha = 0.5,align ="center")
plt.title(f"Prevalence of hypertension among US adults (18+)",fontsize=15)
plt.xlabel("Race", fontsize= 10)
plt.ylabel("Average %",fontsize= 10)
plt.xticks(fontsize=10, rotation=90)
plt.ylim(0,100)
plt.savefig("Images/bw_Prevalence of hypertension among US adults.png")
plt.show()


# In[136]:


file= "datafiles/IMR_by_race.csv"
IMR_df=pd.read_csv(file)
IMR_df


# In[137]:


IMR_df=IMR_df.set_index("Location")


# In[138]:


list_IMR_rate=list(IMR_df.loc["United States"])


# In[139]:


label = ['American Indian', 'Asian and Pacific Islander', 'African American', 'Hispanic or Latino', 'Non-Hispanic White']
bar_width=0.3
plt.figure(figsize=(10,8))
r1=np.arange(len(label))
r2=[x + bar_width for x in r1]
r3=[x + bar_width for x in r2]
plt.bar(r1, bw_race_y_axis,color = "#7fe5b9",width=bar_width,edgecolor='white', label='Preterm Baby')
plt.bar(r2,list_hypertension, color = "#bde592",width=bar_width,edgecolor='white', label='Hypertension')
plt.bar(r3,list_overweight, color = "#ffba50", width=bar_width,edgecolor='white', label='Overweight')
plt.plot(label, list_IMR_rate,label = "IMR Rate",color = "#fc6060", marker='o', linestyle='dashed',linewidth=2, markersize=12)
plt.title(f"Infant Mortality Rate Factors by Race",fontsize=15)
plt.xlabel("Race", fontsize= 10)
plt.ylabel("Rate (Average %)",fontsize= 10)
plt.xticks([r+bar_width for r in range(len(label))],['American Indian', 'Asian and Pacific Islander', 'African American', 'Hispanic or Latino', 'Non-Hispanic White'], fontsize=10, rotation=90)
plt.ylim(0,100)
plt.legend()
plt.savefig("Images/bw_IMR Factors by Race.png", bbox_inches="tight")
plt.show()


# In[ ]:





# In[140]:


#read census poverty data
poverty_df = pd.read_csv('datafiles/2007_2016_poverty.csv')


# In[141]:


poverty_df.head()


# In[142]:


#only keep needed columns
poverty_df = poverty_df[["Year","County ID","State / County Name","All Ages in Poverty Percent"]]


# In[143]:


poverty_df.head()


# In[144]:


#group by the County ID
poverty_group = poverty_df.groupby(["County ID"]).mean()


# In[145]:


poverty_group.head()


# In[146]:


#read table of lat/lng cooridnates for counties
location_df = pd.read_csv('datafiles/2017_counties.csv', encoding="ISO-8859-1")


# In[147]:


location_df = location_df.rename(columns={'GEOID': "County ID","INTPTLAT": "Lat","INTPTLONG": "Lng"})


# In[148]:


location_df.head()


# In[149]:


#combine poverty data with county location data
merge_table = pd.merge(poverty_df, location_df, on="County ID", how="left")


# In[150]:


merge_table.head()


# In[151]:


#group by County ID
merge_group = merge_table.groupby(["County ID"]).mean()


# In[152]:


merge_group.head()


# In[153]:


#remove rows with missing values
merge_group = merge_group.dropna(how="any")


# In[154]:


merge_group.head()


# In[155]:


#load API key
gmaps.configure(api_key=gkey)


# In[156]:


#construct heat map of poverty levels from 2007-2016
locations = merge_group[["Lat", "Lng"]].astype(float)
poverty_rate = merge_group["All Ages in Poverty Percent"].astype(float)
fig = gmaps.figure()
heat_layer = gmaps.heatmap_layer(locations, weights=poverty_rate, 
                                 dissipating=False, max_intensity=100,
                                 point_radius = .6)


# In[157]:


fig.add_layer(heat_layer)

fig


# In[158]:


#read CDC data on top 15 states for infant death rates
death_rates_state_top_df = pd.read_csv('datafiles/death_rates_state_top.csv')


# In[159]:


death_rates_state_top_df


# In[160]:


#drop location markers on the 15 states
locations_state = death_rates_state_top_df[["Lat", "Lng"]].astype(float)
state_layer = gmaps.symbol_layer(
    locations_state, fill_color='rgba(0, 150, 0, 0.4)',
    stroke_color='rgba(0, 0, 150, 0.4)', scale=4)

#state_layer.markers[10].scale=20



fig = gmaps.figure()
fig.add_layer(state_layer)

fig


# In[161]:


#combine poverty heatmap with state location
fig = gmaps.figure()
fig.add_layer(heat_layer)
fig.add_layer(state_layer)
fig


# In[162]:


#read CDC data on death rates per county for 2006-2017
death_rates_county = pd.read_csv('datafiles/death_rates.csv')


# In[163]:


death_rates_county.head()


# In[164]:


death_rates_county = death_rates_county.rename(columns={"County Code": "County ID"})


# In[165]:


death_rates_county.head()


# In[166]:


#merge CDC data on death rates per county with poverty and county location data
regress_df = pd.merge(death_rates_county, merge_group, on="County ID", how="left")


# In[167]:


regress_df.head()


# In[168]:


#remove all rows with missing values
regress_df = regress_df.dropna(how="any")


# In[169]:


regress_df.head()


# In[170]:


#define x and y axis for regression analsys
x_axis = regress_df["All Ages in Poverty Percent"]
y_axis = regress_df["Death Rate"]


# In[171]:


(slope, intercept, _, _, _) = linregress(x_axis, y_axis)
fit = slope * x_axis + intercept


# In[172]:


#calculate statistical values
slope, intercept, r_value, p_value, std_err = linregress(x_axis, y_axis)


# In[173]:


#perform linear regression of death rate versus poverty
fig, ax = plt.subplots()

fig.suptitle("Death Rate v Poverty 2007-2016", fontsize=16, fontweight="bold")

ax.set_xlim(0,35)
ax.set_ylim(0,15)

ax.set_xlabel("Poverty Rate (%)")
ax.set_ylabel("Death Rate (per 1000)")

ax.plot(x_axis, y_axis, linewidth=0, marker='o')
ax.plot(x_axis, fit, 'b--')
plt.savefig("Images/deathrateVpoverty_linregress.png")
plt.show()


# In[174]:


p_value


# In[175]:


#start health insurance analysis
insurance_df = pd.read_csv("datafiles/insurance.csv")


# In[176]:


insurance_df.head()


# In[177]:


death_rates_state = pd.read_csv("datafiles/death_rates_state.txt", delimiter="\t")


# In[178]:


death_rates_state = death_rates_state[["State","Death Rate"]]


# In[179]:


death_rates_state = death_rates_state.dropna(how="any")


# In[180]:


death_rates_state.head()


# In[181]:


insurance_group = insurance_df.groupby("State").mean()


# In[182]:


insurance_group.head()


# In[183]:


insurance_merge = pd.merge(death_rates_state, insurance_group, on="State", how="left")


# In[184]:


insurance_merge.head()


# In[185]:


x_axis = insurance_merge["Total"]
y_axis = insurance_merge["Death Rate"]


# In[186]:


#graph the death rate versus total rate of insurance for all states
plt.scatter(x_axis, y_axis)
plt.ylabel("Death Rate (per 1000)")
plt.xlabel("Rate of Insurance Coverage (%)")
plt.title("Death Rate v Total Insurance Rate")
plt.savefig("Images/DeathRate_v_TotalInsurance.png")


# In[187]:


(slope, intercept, _, _, _) = linregress(x_axis, y_axis)
fit = slope * x_axis + intercept
slope, intercept, r_value, p_value, std_err = linregress(x_axis, y_axis)


# In[188]:


p_value


# In[189]:


x2_axis = insurance_merge["Public"]


# In[190]:


insurance_merge = insurance_merge.dropna(how="any")


# In[191]:


insurance_merge.dtypes


# In[192]:


#plot death rate versus rate of public insurance
(slope, intercept, _, _, _) = linregress(x2_axis, y_axis)
fit = slope * x2_axis + intercept
slope, intercept, r_value, p_value, std_err = linregress(x2_axis, y_axis)

fig, ax = plt.subplots()

fig.suptitle("", fontsize=16, fontweight="bold")

ax.set_xlim(15,55)
ax.set_ylim(4,10)

ax.plot(x2_axis, y_axis, linewidth=0, marker='o')
ax.plot(x2_axis, fit, 'b--')
plt.scatter(x2_axis, y_axis)
plt.ylabel("Death Rate (per 1000)")
plt.xlabel("Rate of Public Insurance Coverage (%)")
plt.title("Death Rate v Public Insurance Rate")
plt.savefig("Images/DeathRate_v_PublicInsurance.png")
plt.show()


# In[193]:


p_value


# In[194]:


x3_axis = insurance_merge["Private"]


# In[195]:


#plot death rate versus rate of private insurance
(slope, intercept, _, _, _) = linregress(x3_axis, y_axis)
fit = slope * x3_axis + intercept
slope, intercept, r_value, p_value, std_err = linregress(x3_axis, y_axis)

fig, ax = plt.subplots()

fig.suptitle("", fontsize=16, fontweight="bold")

ax.set_xlim(40,80)
ax.set_ylim(4,10)

ax.plot(x3_axis, y_axis, linewidth=0, marker='o')
ax.plot(x3_axis, fit, 'b--')
plt.scatter(x3_axis, y_axis)
plt.ylabel("Death Rate (per 1000)")
plt.xlabel("Rate of Private Insurance Coverage (%)")
plt.title("Death Rate v Private Insurance Rate")
plt.savefig("Images/DeathRate_v_PrivateInsurance")
plt.show()


# In[196]:


p_value


# In[197]:


#Start new analysis on top ten and bottom ten african american counties regarding death rate
AfricanAmerican20 = pd.read_csv("datafiles/AfricanAmerican20.csv")


# In[198]:


AfricanAmerican20


# In[199]:


AfricanAmerican20 = AfricanAmerican20.rename(columns={"County Code": "County ID"})


# In[200]:


african_merge = pd.merge(AfricanAmerican20, merge_group, on="County ID", how="left")


# In[201]:


african_merge = african_merge[["County ID","Death Rate","All Ages in Poverty Percent"]]


# In[202]:


african_merge


# In[203]:


x_axis = african_merge["All Ages in Poverty Percent"]
y_axis = african_merge["Death Rate"]


# In[204]:


african_merge.set_index("County ID").plot.bar("Death Rate")


# In[205]:


black_counties = pd.read_csv("datafiles/black_counties.csv")


# In[206]:


black_counties.head()


# In[207]:


black_merge = pd.merge(black_counties, merge_group, on="County ID", how="left")


# In[208]:


black_merge = black_merge.dropna(how="any")


# In[209]:


black_merge.head()


# In[210]:


x_axis = black_merge["All Ages in Poverty Percent"]
y_axis = black_merge["Death Rate"]
plt.scatter(x_axis, y_axis)


# In[211]:


#plot death rate versus poverty rate for African Americans
(slope, intercept, _, _, _) = linregress(x_axis, y_axis)
fit = slope * x_axis + intercept
slope, intercept, r_value, p_value, std_err = linregress(x_axis, y_axis)

fig, ax = plt.subplots()

fig.suptitle("", fontsize=16, fontweight="bold")

ax.set_xlim(4,31)
ax.set_ylim(5,18)

# ax.set_xlabel("Poverty Rate (%)")
# ax.set_ylabel("Death Rate (%)")

ax.plot(x_axis, y_axis, linewidth=0, marker='o')
ax.plot(x_axis, fit, 'b--')
plt.scatter(x_axis, y_axis)
plt.ylabel("Death Rate (per 1000)")
plt.xlabel("Poverty Rate (%)")
plt.title("Death Rate v Poverty Rate for African Americans")
plt.savefig("Images/DeathRate_v_AfricanAmericanPoverty.png")
plt.show()


# In[212]:


p_value


# In[213]:


white_counties = pd.read_csv("datafiles/white_counties.csv")


# In[214]:


white_counties.head()


# In[215]:


white_merge = pd.merge(white_counties, merge_group, on="County ID", how="left")


# In[216]:


white_merge = white_merge.dropna(how="any")


# In[217]:


#plot death rate versus poverty rate for Whites
x_axis = white_merge["All Ages in Poverty Percent"]
y_axis = white_merge["Death Rate"]

(slope, intercept, _, _, _) = linregress(x_axis, y_axis)
fit = slope * x_axis + intercept
slope, intercept, r_value, p_value, std_err = linregress(x_axis, y_axis)

fig, ax = plt.subplots()

fig.suptitle("", fontsize=16, fontweight="bold")

ax.set_xlim(4,35)
ax.set_ylim(2,8)

ax.plot(x_axis, y_axis, linewidth=0, marker='o')
ax.plot(x_axis, fit, 'b--')
plt.scatter(x_axis, y_axis)
plt.ylabel("Death Rate (per 1000)")
plt.xlabel("Poverty Rate (%)")
plt.title("Death Rate v Poverty Rate for Whites")
plt.savefig("Images/DeathRate_v_PovertyRateWhites.png")
plt.show()


# In[218]:


p_value


# In[219]:


#Heatmap with poverty per county and top ten counties deathrate
regress_df = regress_df.sort_values(by=["Death Rate"], ascending=False)


# In[220]:


regress_df = regress_df.reset_index()


# In[221]:


regress_df.head()


# In[222]:


regress_df = regress_df.iloc[0:10]


# In[223]:


regress_df


# In[224]:


regress_df.to_csv("datafiles/low_IMR_county.csv")


# In[225]:


gmaps.configure(api_key=gkey)
#construct heat map of poverty levels from 2007-2016
locations = merge_group[["Lat", "Lng"]].astype(float)
poverty_rate = merge_group["All Ages in Poverty Percent"].astype(float)
fig = gmaps.figure()
heat_layer = gmaps.heatmap_layer(locations, weights=poverty_rate, 
                                 dissipating=False, max_intensity=100,
                                 point_radius = .6)

#drop location markers on the 10 counties
locations_county = regress_df[["Lat", "Lng"]].astype(float)
county_layer = gmaps.symbol_layer(
    locations_county, fill_color='rgba(0, 150, 0, 0.4)',
    stroke_color='rgba(0, 0, 150, 0.4)', scale=4)

fig = gmaps.figure()
fig.add_layer(heat_layer)
fig.add_layer(county_layer)
fig


# In[226]:


#
# Logan Caldwell
#

get_ipython().run_line_magic('matplotlib', 'notebook')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

mothers_age_state_csv = "datafiles/mothers_age_state_grouped.csv"

df_mothers_age_state = pd.read_csv(mothers_age_state_csv)

df_mothers_age_state.head(15)


# In[227]:


df_mothers_age_state.describe()

### Dropna working correctly here??
df_mothers_age_state.dropna(axis=0, how="any")
df_mothers_age_state["Age of Mother"].unique()


# In[228]:


df_mothers_age_state.mean()


# In[229]:


ages_15_19 = df_mothers_age_state[df_mothers_age_state["Age of Mother"] == "15-19 years"]
ages_20_24 = df_mothers_age_state[df_mothers_age_state["Age of Mother"] == "20-24 years"]
ages_25_29 = df_mothers_age_state[df_mothers_age_state["Age of Mother"] == "25-29 years"]
ages_30_34 = df_mothers_age_state[df_mothers_age_state["Age of Mother"] == "30-34 years"]
ages_35_39 = df_mothers_age_state[df_mothers_age_state["Age of Mother"] == "35-39 years"]
ages_40_44 = df_mothers_age_state[df_mothers_age_state["Age of Mother"] == "40-44 years"]

ages_15_19["Death Rate"] = ages_15_19["Death Rate"].str.replace("\s*\(Unreliable\)", "")
ages_20_24["Death Rate"] = ages_20_24["Death Rate"].str.replace("\s*\(Unreliable\)", "")
ages_25_29["Death Rate"] = ages_25_29["Death Rate"].str.replace("\s*\(Unreliable\)", "")
ages_30_34["Death Rate"] = ages_30_34["Death Rate"].str.replace("\s*\(Unreliable\)", "")
ages_35_39["Death Rate"] = ages_35_39["Death Rate"].str.replace("\s*\(Unreliable\)", "")
ages_40_44["Death Rate"] = ages_40_44["Death Rate"].str.replace("\s*\(Unreliable\)", "")


# In[230]:


ages_15_19["Death Rate"] = ages_15_19["Death Rate"].astype(float)
ages_20_24["Death Rate"] = ages_20_24["Death Rate"].astype(float)
ages_25_29["Death Rate"] = ages_25_29["Death Rate"].astype(float)
ages_30_34["Death Rate"] = ages_30_34["Death Rate"].astype(float)
ages_35_39["Death Rate"] = ages_35_39["Death Rate"].astype(float)
ages_40_44["Death Rate"] = ages_40_44["Death Rate"].astype(float)

ages_15_19_IMR_mean = (ages_15_19["Death Rate"].mean())
ages_20_24_IMR_mean = (ages_20_24["Death Rate"].mean())
ages_25_29_IMR_mean = (ages_25_29["Death Rate"].mean())
ages_30_34_IMR_mean = (ages_30_34["Death Rate"].mean())
ages_35_39_IMR_mean = (ages_35_39["Death Rate"].mean())
ages_40_44_IMR_mean = (ages_40_44["Death Rate"].mean())

IMR_rate_means_by_age_list = [ages_15_19_IMR_mean,ages_20_24_IMR_mean,ages_25_29_IMR_mean,ages_30_34_IMR_mean,ages_35_39_IMR_mean,ages_40_44_IMR_mean]
IMR_rate_means_by_age_list


# In[231]:


df_mothers_age_state_means = df_mothers_age_state.mean()
df_mothers_age_state_means.head()


# In[232]:


df_mothers_age_state.set_index("State")
df_mothers_age_state_grouped = df_mothers_age_state.groupby(by="State", group_keys=True,)
df_mothers_age_state.mean()


# In[233]:


age_ticks = [0,1,2,3,4,5]
age_ranges_list = ["15-19", "20-24", "25-29", "30-34", "35-39", "40-44"]

x=[0,1,2,3,4,5]


# In[234]:


m_age_IMR_plot = plt.scatter(age_ranges_list, IMR_rate_means_by_age_list, )


# In[235]:


plt.title("Mother's Age and Infant Mortality Rate")
plt.xlabel("Age of Mother")
plt.ylabel("Infant Mortality Rate (%)")
plt.grid()
# plt.legend(loc="best", labels=age_ranges_list)
plt.xlim(-1,6)
plt.ylim(0, max(IMR_rate_means_by_age_list)+2)
plt.tight_layout()


# In[237]:


plt.savefig("Images/IMR_and_age_of_mother_plot")


# In[236]:


plt.show()


# In[ ]:




