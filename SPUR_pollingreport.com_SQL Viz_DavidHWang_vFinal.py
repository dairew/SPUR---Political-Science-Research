
# DONE in JUPYTER - May be possible translation errors when downloading as a straight python file

# In[2]:

get_ipython().magic(u'load_ext sql')


# In[5]:

get_ipython().magic(u'sql mysql://root:PRIVATE@localhost:3306/spur?charset=utf8')


# In[50]:

get_ipython().magic(u'sql show tables')


# In[51]:

get_ipython().magic(u'sql describe pollingreport')


# In[52]:

# Import packages

import numpy as np
import matplotlib.pyplot as pyplot
import pandas as pd


# In[75]:

# Issue queries and store in result
# Create dataframes from result

# Get average percentage for each issue by year
issue_yoy_result = get_ipython().magic(u"sql SELECT Year, Issue, ROUND(AVG(Percentage),2) AS 'Average Percent' FROM pollingreport GROUP BY Issue, Year ORDER BY Year DESC, ROUND(SUM(Percentage),2) DESC;")
issue_yoy_df = issue_yoy_result.DataFrame()

# 10 least mentioned topics
least_ten_result = get_ipython().magic(u"sql SELECT Year, Issue, SUM(Percentage) AS 'Total Percentage Points' FROM pollingreport GROUP BY Issue ORDER BY SUM(Percentage) ASC LIMIT 10;")
least_ten_df = least_ten_result.DataFrame()

# TOP TOPICS TIME SERIES
## econ
econ_result = get_ipython().magic(u"sql SELECT Year, ROUND(AVG(Percentage),2) AS 'Average Percent' FROM pollingreport WHERE Issue = 'econ' GROUP BY Issue, Year ORDER BY Year ASC;")
econ_df = econ_result.DataFrame()

## terror
terror_result = get_ipython().magic(u"sql SELECT Year, ROUND(AVG(Percentage),2) AS 'Average Percent' FROM pollingreport WHERE Issue = 'terror' GROUP BY Issue, Year ORDER BY Year ASC;")
terror_df = terror_result.DataFrame()

## immigration
immigration_result = get_ipython().magic(u"sql SELECT Year, ROUND(AVG(Percentage),2) AS 'Average Percent' FROM pollingreport WHERE Issue = 'immigration' GROUP BY Issue, Year ORDER BY Year ASC;")
immigration_df = immigration_result.DataFrame()

## race
race_result = get_ipython().magic(u"sql SELECT Year, ROUND(AVG(Percentage),2) AS 'Average Percent' FROM pollingreport WHERE Issue = 'race' GROUP BY Issue, Year ORDER BY Year ASC;")
race_df = race_result.DataFrame()

## education
education_result = get_ipython().magic(u"sql SELECT Year, ROUND(AVG(Percentage),2) AS 'Average Percent' FROM pollingreport WHERE Issue = 'education' GROUP BY Issue, Year ORDER BY Year ASC;")
education_df = education_result.DataFrame()

## health
health_result = get_ipython().magic(u"sql SELECT Year, ROUND(AVG(Percentage),2) AS 'Average Percent' FROM pollingreport WHERE Issue = 'health' GROUP BY Issue, Year ORDER BY Year ASC;")
health_df = health_result.DataFrame()

## budget
budget_result = get_ipython().magic(u"sql SELECT Year, ROUND(AVG(Percentage),2) AS 'Average Percent' FROM pollingreport WHERE Issue = 'budget' GROUP BY Issue, Year ORDER BY Year ASC;")
budget_df = budget_result.DataFrame()

## unsure
unsure_result = get_ipython().magic(u"sql SELECT Year, ROUND(AVG(Percentage),2) AS 'Average Percent' FROM pollingreport WHERE Issue = 'unsure' GROUP BY Issue, Year ORDER BY Year ASC;")
unsure_df = unsure_result.DataFrame()




# In[76]:

race_df
    


# In[84]:

get_ipython().magic(u'matplotlib inline')
plt = pyplot
plt.figure(figsize=(15,7))
plt.plot(econ_df['Year'], econ_df['Average Percent'], label = "Economy/Jobs")
plt.plot(terror_df['Year'], terror_df['Average Percent'], label = "Terrorism")
plt.plot(immigration_df['Year'], immigration_df['Average Percent'], label = "Immigration")
plt.plot(race_df['Year'], race_df['Average Percent'], label = "Race")
plt.plot(education_df['Year'], education_df['Average Percent'], label = "Education")
plt.plot(health_df['Year'], health_df['Average Percent'], label = "Health care")
plt.plot(budget_df['Year'], budget_df['Average Percent'], label = "Budget/Deficit")
plt.xlabel("Years")
plt.ylabel("Average %")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()


# In[85]:

# Least talked about topics:
get_ipython().magic(u"sql SELECT Year, Issue, SUM(Percentage) AS 'Total Percentage Points' FROM pollingreport GROUP BY Issue ORDER BY SUM(Percentage) ASC LIMIT 10;")

## These very one-off issues represent 


# In[ ]:



