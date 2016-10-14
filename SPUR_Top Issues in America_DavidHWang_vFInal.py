# DONE in JUPYTER

# # Top Issues Through History, According to Democrats and Republicans

# ## Using party manifesto transcripts from 1840-2012

# In[1]:

import os
import pandas as pd
import nltk
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

plt.rcParams['figure.figsize'] = (15, 5)

# Get initial file path
file_path_d = '/Users/DHW/Dropbox/spur-david/Corpus/party_democrat/'
file_path_r = '/Users/DHW/Dropbox/spur-david/Corpus/party_republican/'

# Set file variables
file_names_d = []
file_names_r = []
democrats = []
republicans = []

# Create dictionary to hold string versions of all strings
dem_dict_string = {}
rep_dict_string = {}
year_dem = 1840
year_rep = 1856

# Go through main file and append democratic speeches to 'democrats' and republican speeches to 'republicans'
for file in os.listdir('/Users/DHW/Dropbox/spur-david/Corpus/party_democrat/'):
    # Attach file name to file path!
    f = open(file_path_d+file, 'r')
    f = f.read()
    democrats.append(f)
    dem_dict_string[year_dem] = f
    year_dem+=4
    
## Normalize with republican files by starting from 1856
democrats = democrats[4:]

for file in os.listdir('/Users/DHW/Dropbox/spur-david/Corpus/party_republican/'):
    f = open(file_path_r+file, 'r')
    f = f.read()
    republicans.append(f)
    rep_dict_string[year_rep] = f
    year_rep+=4



# ### Clean party manifesto data through tokenization and lemmatization

# In[2]:

# Remove all punctuation in democrats
import re

# Create dictionarys, keys: year, values: party manifesto for that year
dem_dict = {}
dem_dict_sent = {}
year = 1856

# Iterate through democrats to grab speeches
for speech in democrats:
    # decode string to be unicode friendly
    speech = speech.decode('utf-8')
    # Tokenize each speech into words and lowercase
    # Tokenize each speech into sentences and lowercase
    unreg_word = nltk.word_tokenize(speech.lower())
    d_word = [w for w in unreg_word if w.isalnum()]
    
    # lemmatize sentences and words
    wnl = nltk.WordNetLemmatizer()
    d_word_lemmatized =  [wnl.lemmatize(t) for t in d_word]

    dem_dict[year] = d_word_lemmatized
    year+=4


# In[3]:

# Remove all punctuation in republicans

# Create dictionarys, keys: year, values: party manifesto for that year
rep_dict = {}
rep_dict_sent = {}
year = 1856

# Iterate through democrats to grab speeches
for speech in republicans:
    # decode string to be unicode friendly
    speech = speech.decode('utf-8')
    # Tokenize each speech into words and lowercase
    # Tokenize each speech into sentences and lowercase
    unreg_word = nltk.word_tokenize(speech.lower())
    r_word = [w for w in unreg_word if w.isalnum()]

    # lemmatize sentences and words
    wnl = nltk.WordNetLemmatizer()
    r_word_lemmatized =  [wnl.lemmatize(t) for t in r_word]

    rep_dict[year] = r_word_lemmatized
    year+=4


# ### Create Frequency Distribution for 50 Most Common Words in 1856 and 2012

# In[4]:

def rep_common_words(year, n):
    text_rep = nltk.Text(rep_dict[year])
    fdist_rep = text_rep.vocab()
    text_rep_nostop = remove_stopwords(text_rep, fdist_rep.hapaxes())
    fdist_rep_nostop = text_rep_nostop.vocab()
    print 'Republican',year,n, 'Most Common Words\n'
    print fdist_rep_nostop.most_common(n)
    return fdist_rep_nostop
    
def dem_common_words(year, n):
    text_dem = nltk.Text(dem_dict[year])
    fdist_dem = text_dem.vocab()
    text_dem_nostop = remove_stopwords(text_dem,fdist_dem.hapaxes())
    fdist_dem_nostop = text_dem_nostop.vocab()
    print 'Democrat',year,n, 'Most Common Words\n'
    print fdist_dem_nostop.most_common(n)
    return fdist_dem_nostop
    
def remove_stopwords(text, hapaxes):
    stopwords = nltk.corpus.stopwords.words('english')
    content = [w.lower() for w in text 
               if w.lower() not in stopwords # w should not be in NLTK stopwords
                   and w.lower() not in hapaxes] # w should have frequency > 1 
    return nltk.Text(content)


# In[5]:

rep_common_words(1880,50).plot(50, cumulative=False)

dem_common_words(1880,50).plot(50, cumulative=False)


# In[6]:

rep_common_words(2012,50).plot(50, cumulative=False)
dem_common_words(2012,50).plot(50, cumulative=False)


# ### Multi-Series Line Plot of Most Important Issues By Party, throughout History

# In[7]:

import requests
import json
import urllib

def analyzeText(text):
    api_url = "http://gateway-a.watsonplatform.net/calls/text/TextGetRankedTaxonomy"
    api_key = '83c249378fecccd16552ef1b44b753d39e3c61c7'
    headers = {
        "Accept": "application/json"
    }
    parameters = {
        'outputMode': 'json',
        'apikey' : api_key,
        'text': text
    }
    resp = requests.post(api_url, params=parameters, headers=headers)
    data = json.loads(resp.text)
    
    return data


# In[8]:

import requests # This command allows us to fetch URLs
from lxml import html # This module will allow us to parse the returned HTML/XML
import pandas # To create a dataframe

def word_extract(url):
    response = requests.get(url) 
    doc = html.fromstring(response.text) 
    
    wordNode = doc.find('.//*[@id="thesaurusentry"]/div')


    words = wordNode.findall('./div/a/h3')
    data = [(a.text_content()) for a in words]
    return data


# In[9]:

econ_words =  word_extract("http://www.macmillandictionary.com/us/thesaurus-category/american/economy-and-economics")
war_words = word_extract("http://www.macmillandictionary.com/us/thesaurus-category/american/fighting-in-a-war-and-relating-to-fighting-in-a-war")
health_words = word_extract("http://www.macmillandictionary.com/us/thesaurus-category/american/help-with-social-problems")
law_words = word_extract("http://www.macmillandictionary.com/us/thesaurus-category/american/the-law-laws-and-parts-of-laws")
education_words = word_extract("http://www.macmillandictionary.com/us/thesaurus-category/american/general-words-for-teaching")
immigration_words = word_extract("http://www.macmillandictionary.com/us/thesaurus-category/american/moving-to-and-living-in-a-different-country")


# In[10]:

# create topic dictionaries and lists of topic words

dem_economy_dict = {}
rep_economy_dict = {}
all_economy_dict = {}
economy_words = econ_words

dem_war_dict = {}
rep_war_dict = {}
all_war_dict = {}
war_words = war_words

dem_healthcare_dict = {}
rep_healthcare_dict = {}
all_healthcare_dict = {}
healthcare_words = health_words

dem_education_dict = {}
rep_education_dict = {}
all_education_dict = {}
education_words = education_words

dem_immigration_dict = {}
rep_immigration_dict = {}
all_immigration_dict = {}
immigration_words = immigration_words

dem_environment_dict = {}
rep_environment_dict = {}
all_environment_dict = {}
environment_words = ['environment', "climate"]

dem_regu_dict = {}
rep_regu_dict = {}
all_regu_dict = {}
regu_words = law_words

dem_women_dict = {}
rep_women_dict = {}
all_women_dict = {}
women_words = ["women", "reproductive", "sexism", "sexist", "feminism"]


# In[11]:

# get all the years

raw_years = range(1856,2013)
years = raw_years[0::4]


# In[12]:

# method for making creating topic dictionaries

def create_topic_dict(party_dict, topic_dict, topic_words, years):    
    for i in years:
        count = 0.0
        for word in party_dict[i]:
            if (word in topic_words):
                count += 1.0
        topic_dict[i] = (count/float(len(party_dict[i]))*100)



# In[13]:

# DEMOCRATS
create_topic_dict(dem_dict, dem_economy_dict, economy_words, years)
create_topic_dict(dem_dict, dem_war_dict, war_words, years)
create_topic_dict(dem_dict, dem_healthcare_dict, healthcare_words, years)
create_topic_dict(dem_dict, dem_education_dict, education_words, years)
create_topic_dict(dem_dict, dem_regu_dict, regu_words, years)
create_topic_dict(dem_dict, dem_immigration_dict, immigration_words, years)

# Republicans
create_topic_dict(rep_dict, rep_economy_dict, economy_words, years)
create_topic_dict(rep_dict, rep_war_dict, war_words, years)
create_topic_dict(rep_dict, rep_healthcare_dict, healthcare_words, years)
create_topic_dict(rep_dict, rep_education_dict, education_words, years)
create_topic_dict(rep_dict, rep_regu_dict, regu_words, years)
create_topic_dict(rep_dict, rep_immigration_dict, immigration_words, years)


# In[14]:

# test create topic dict method


print rep_economy_dict[2008]


# In[15]:

# Import packages

import numpy as np
import matplotlib.pyplot as pyplot
plt.rcParams['figure.figsize'] = (15, 10)


# In[16]:

# Democrats Top Issues Through History
plt = pyplot
plt.plot(years, dem_economy_dict.values(), label="Economy")
plt.plot(years, dem_war_dict.values(), label="War")
plt.plot(years, dem_healthcare_dict.values(), label="Healthcare")
plt.plot(years, dem_education_dict.values(), label="Education")
plt.plot(years, dem_regu_dict.values(), label="Regulation")
plt.plot(years, dem_immigration_dict.values(), label="Immigration")
plt.xlabel("Years")
plt.ylabel("Mention Frequency %")
plt.legend()
plt.grid()


# In[17]:

# Republicans Top Issues Through History
plt = pyplot
plt.plot(years, rep_economy_dict.values(), label="Economy")
plt.plot(years, rep_war_dict.values(), label="War")
plt.plot(years, rep_healthcare_dict.values(), label="Healthcare")
plt.plot(years, rep_education_dict.values(), label="Education")
plt.plot(years, rep_regu_dict.values(), label="Regulation")
plt.plot(years, rep_immigration_dict.values(), label="Immigration")
plt.xlabel("Years")
plt.ylabel("Mention Frequency %")
plt.legend()
plt.grid()


# #### Comparing Party Mention Frequencies For Each Topic

# In[18]:

plt.style.use('ggplot')


# In[19]:

fig = plt.figure(figsize=(15,10))
fig2 = plt.figure(figsize=(15,10))
fig3 = plt.figure(figsize=(15,10))

# Create the first subfigure
sub1 = fig.add_subplot(2,2,1)
sub1.set_xlabel('Years')
sub1.set_ylabel('Mention Freq %')
sub1.set_title('Topic: Economy')
sub1.plot(years, rep_economy_dict.values(), label="Republican")
sub1.plot(years, dem_economy_dict.values(), label="Democrat")
sub1.legend()
sub1.grid()

# Create the second subfigure
sub1 = fig.add_subplot(2,2,2)
sub1.set_xlabel('Years')
sub1.set_ylabel('Mention Freq %')
sub1.set_title("Topic: War")
sub1.plot(years, rep_war_dict.values(), label="Republican")
sub1.plot(years, dem_war_dict.values(), label="Democrat")
sub1.legend()
sub1.grid()

# Create the third subfigure
sub2 = fig2.add_subplot(2,2,1)
sub2.set_xlabel('Years')
sub2.set_ylabel('Mention Freq %')
sub2.set_title("Topic: Healthcare")
sub2.plot(years, rep_healthcare_dict.values(), label="Republican")
sub2.plot(years, dem_healthcare_dict.values(), label="Democrat")
sub2.legend()
sub2.grid()

# Create the fourth subfigure
sub2 = fig2.add_subplot(2,2,2)
sub2.set_xlabel('Years')
sub2.set_ylabel('Mention Freq %')
sub2.set_title("Topic: Regulation")
sub2.plot(years, rep_regu_dict.values(), label="Republican")
sub2.plot(years, dem_regu_dict.values(), label="Democrat")
sub2.legend()
sub2.grid()

# Create the fifth subfigure
sub3 = fig3.add_subplot(2,2,1)
sub3.set_xlabel('Years')
sub3.set_ylabel('Mention Freq %')
sub3.set_title("Topic: Education")
sub3.plot(years, rep_education_dict.values(), label="Republican")
sub3.plot(years, dem_education_dict.values(), label="Democrat")
sub3.legend()
sub3.grid()

# Create the sixth subfigure
sub3 = fig3.add_subplot(2,2,2)
sub3.set_xlabel('Years')
sub3.set_ylabel('Mention Freq %')
sub3.set_title("Topic: Immigration")
sub3.plot(years, rep_immigration_dict.values(), label="Republican")
sub3.plot(years, dem_immigration_dict.values(), label="Democrat")
sub3.legend()
sub3.grid()


# In[20]:

# Bar Plot: Republican Overlayed On Democrat
plt.bar(years, dem_economy_dict.values(), label = 'Dem Economy', color = 'b')
plt.bar(years, rep_economy_dict.values(), label = 'Rep Economy', color = 'r')

plt.xlabel('Years')
plt.ylabel('Mention Freq %')
plt.title('Republican Overlayed On Democrat')
plt.legend()

plt.tight_layout()
plt.show()


# In[21]:

# Bar Plot: Democrat Overlayed On Republican
plt.bar(years, rep_economy_dict.values(), label = 'Rep Economy', color = 'r')
plt.bar(years, dem_economy_dict.values(), label = 'Dem Economy', color = 'b')

plt.xlabel('Years')
plt.ylabel('Mention Freq %')
plt.title('Democrat Overlayed On Republican')
plt.legend()

plt.tight_layout()
plt.show()


# # Correlation Matrix

# In[22]:

corr_raw_list = [dem_economy_dict, dem_war_dict, dem_healthcare_dict, dem_education_dict, dem_regu_dict, dem_immigration_dict, rep_economy_dict, rep_war_dict, rep_healthcare_dict, rep_education_dict, rep_regu_dict, rep_immigration_dict]

c_index= ["Dem Economy","Dem War", "Dem Healthcare", "Dem Education", "Dem Regu", "Dem Immigration", "Rep Economy", "Rep War", "Rep Healthcare", "Rep Education", "Rep Regu", "Rep Immigration"]
dummy = [0,0,0,0,0,0,0,0,0,0,0,0,0]
corr_list = []
for x in corr_raw_list:
    temp = []
    for y in corr_raw_list:
        x = pd.Series(x)
        y = pd.Series(y)
        z = float("{0:.2f}".format(x.corr(y)))
        temp.append(z);
    corr_list.append(temp);
c_index.append(" ")


# In[23]:

df = pd.DataFrame({"Dem Econ": dem_economy_dict.values(), 
                   "Dem War": dem_war_dict.values(),
                  "Dem Health": dem_healthcare_dict.values(),
                  "Dem Edu": dem_education_dict.values(),
                  "Dem Regu": dem_regu_dict.values(),
                  "Dem Immi": dem_immigration_dict.values(),
                  "Rep Econ": rep_economy_dict.values(),
                  "Rep War": rep_war_dict.values(),
                  "Rep Health": rep_healthcare_dict.values(),
                  "Rep Edu": rep_education_dict.values(),
                  "Rep Regu": rep_regu_dict.values(),
                  "Rep Immi": rep_immigration_dict.values()})

corrMatrix = df.corr()
corrMatrix


# In[24]:

plotty = corrMatrix.plot(kind='bar', stacked = 'true', colormap = 'spectral')
plotty.legend(loc='center left', bbox_to_anchor=(1, 0.5))

