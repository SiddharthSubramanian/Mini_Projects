
# coding: utf-8

# In[12]:



import os

from IPython.utils.pickleshare import PickleShareDB 

DIRECTORY = 'G:\\Phone backup'  # music source Directory
COPY_DIRECTORY = 'G:\\songs'  # Destination directory

d = path(DIRECTORY)
copy_directory = path(COPY_DIRECTORY)


def transfer():
    
    file_count = 0
    for i in d.walk():
        if i.isfile() and i.endswith('mp3'):
            file_count += 1
            print("Copying %s" % i)
            # i.copy(copy_directory)

    print('Transferred %s files' % file_count)


if __name__ == "__main__":
    transfer()


# In[16]:


get_ipython().system('pip install path')


# In[17]:


get_ipython().system('python -m pip install --upgrade pip')


# In[19]:


get_ipython().system('pip install tpot')


# In[20]:


import glob


# In[26]:


os.chdir('G:\\Phone backup')
from shutil import copyfile


# In[42]:


text = "Sony Phone Backup\Siddhu\My$ Tamil songs\Enakkul_Oruvan\Endi_Ippadi-StarMusiQ.Com.mp3"
text.rfind("\\")
text[0:text.rfind("\\")]


# In[52]:


DIRECTORY = 'G:\\Phone backup'  # music source Directory
COPY_DIRECTORY = 'G:\\songs'


for file_ in glob.glob('**/**/**/**/**'):
    if file_.endswith('.mp3'):
        print(file_[0:file_.rfind("\\")])
    
        print('G:\\Phone backup\\'+file_[0:file_.rfind("\\")] + '.mp3')
        copyfile('G:\\Phone backup\\'+file_[0:file_.rfind("\\")] + '.mp3','G:\\songs')
    


# In[26]:


from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)
CORS(app)


# In[1]:



import requests
from bs4 import BeautifulSoup


def getNews(category):
    newsDictionary = {
        'success': True,
        'category': category,
        'data': []
    }

    try:
        htmlBody = requests.get('https://www.inshorts.com/en/read/' + category)
    except requests.exceptions.RequestException as e:
        newsDictionary['success'] = False
        newsDictionary['errorMessage'] = str(e.message)
        return newsDictionary

    soup = BeautifulSoup(htmlBody.text, 'lxml')
    newsCards = soup.find_all(class_='news-card')
    if not newsCards:
        newsDictionary['success'] = False
        newsDictionary['errorMessage'] = 'Invalid Category'
        return newsDictionary

    for card in newsCards:
        try:
            title = card.find(class_='news-card-title').find('a').text
        except AttributeError:
            title = None

        try:
            imageUrl = card.find(
                class_='news-card-image')['style'].split("'")[1]
        except AttributeError:
            imageUrl = None

        try:
            url = ('https://www.inshorts.com' + card.find(class_='news-card-title')
                   .find('a').get('href'))
        except AttributeError:
            url = None

        try:
            content = card.find(class_='news-card-content').find('div').text
        except AttributeError:
            content = None

        try:
            author = card.find(class_='author').text
        except AttributeError:
            author = None

        try:
            date = card.find(clas='date').text
        except AttributeError:
            date = None

        try:
            time = card.find(class_='time').text
        except AttributeError:
            time = None

        try:
            readMoreUrl = card.find(class_='read-more').find('a').get('href')
        except AttributeError:
            readMoreUrl = None

        newsObject = {
            'title': title,
            'imageUrl': imageUrl,
            'url': url,
            'content': content,
            'author': author,
            'date': date,
            'time': time,
            'readMoreUrl': readMoreUrl
        }

        newsDictionary['data'].append(newsObject)

    return newsDictionary


# In[8]:


jsonify(getNews(request.form['category']))


# In[3]:


def news():
    if request.method == 'POST':
        return jsonify(getNews(request.form['category']))
    elif request.method == 'GET':
        return jsonify(getNews(request.args.get('category')))


# In[30]:


df = pd.DataFrame(columns=['content','url'])


# In[110]:


news =  getNews('national')
# national //Indian National News
# business
# sports
# world
# politics
# technology
# startup
# entertainment
# miscellaneous
# hatke // Unconventional
# science
# automobile


# In[111]:


df = []
for idx , i in enumerate(list(news['data'])):
    temp = i 
#     print(temp)
#     df.append((temp['content'],temp['readMoreUrl']))
    t3 = pd.DataFrame({'Content': temp['content'], 'Original Url': temp['readMoreUrl']}, index=[0])
    df_news = df_news.append(t3,ignore_index= True)
#     df_news.append(pd.DataFrame({'Content': temp['content'], 'Original Url': temp['readMoreUrl']}, index=[0]), ignore_index=True)


# In[115]:


df_news.to_pickle('E:\\news.pkl')


# In[114]:


import pickle
pickle.dump(file=df_news,)


# In[ ]:


df_news  = pd.DataFrame(df, columns=('Content', 'Original Url'))
#     content = temp['content']
#     original news = temp['readMoreUrl']


# In[53]:


df_news


# In[38]:


temp


# In[24]:


news['data']

