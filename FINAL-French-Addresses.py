#!/usr/bin/env python
# coding: utf-8

from bs4 import BeautifulSoup
import pickle
import pandas as pd
import re
import spacy
from collections import OrderedDict
import requests
import os
from os.path import join
import numpy as np
from tqdm import tqdm

nlp = spacy.load('fr_core_news_sm')

firstnames = {}
genders = {'féminin':'fem','masculin':'masc'}
with open('dataset-firstnames.html','r') as htmlfp:
    soup = BeautifulSoup(htmlfp,"lxml")
    dds = soup.find_all('dd')
for dd in dds:
    try: 
        firstname = dd.a.text.lower()
        gender = dd.span['title'].split()[-1]
        firstnames[firstname]=genders[gender]
    except:
        pass
print('[PREPARE] Analysis successful,',len(firstnames),'found: stored in [firstnames]')


with open('dataset-adjectives.pkl','rb') as fp:
    adjectives = pickle.load(fp)
    adj_high_priority = [a for a,p in adjectives if p==1]
    adj_low_priority = [a for a,p in adjectives if p>1]
print('[PREPARE] Analysis successful,',len(adjectives),'found: stored in [adjectives]')

with open('dataset-nouns.pkl','rb') as fp:
    nouns = pickle.load(fp)
print('[PREPARE] Analysis successful,',len(nouns),'found: stored in [nouns]')

with open('dataset-grade.txt','r') as fp:
    lines = fp.readlines()
    gds=set()
    for line in lines:
        gds.add(line.strip().lower())
grades = list(gds)
print('[PREPARE] Analysis successful,',len(grades),'found: stored in [grades]')


dfLemmas = pd.read_excel('france-street-category.DB2.xls',
                   sheet_name='france-street-category',
                   usecols='A:I',
                   names=['raw_regex','lemma','signifier_lvl_0','signifier_lvl_1','signifier_lvl_2','signifier_lvl_3','signifier_common','etymology_language','etymology_region'])

dfLemmas = dfLemmas.sort_values(by='raw_regex', ascending=False)
regexes = []
for index, row in dfLemmas.iterrows():
    regex = re.compile(r'\b'+ row['raw_regex'] + r'\b', re.IGNORECASE | re.DOTALL | re.UNICODE)
    regexes.append(regex)
dfLemmas = dfLemmas.assign(regex=regexes)
print('[PREPARE] first row of the dataframe shall last alphabetical longer one:',dfLemmas.iloc[0]['lemma'])


def bestCandidate(name):
    matches = []
    for index,row in dfLemmas.iterrows():
        match = row['regex'].search(name)
        # if regex match on a name, append row index in dataframe, position and length of the match
        # prioritize over the closer to beginning and longer match
        if match is not None:
            matches.append((index, match.span()[0], match.span()[1]-match.span()[0]))
    try:
        index,start,length = sorted(matches, key=lambda tup: tup[1],reverse=False)[0]
        return dfLemmas.loc[index]
    except IndexError:
        return None


class Odonym(object):
    def __init__(self,name,lemma,lieuDit):
        self.df = None
        self.name = name
        self.lemma = lemma
        self.original = None
        self.lieuDit = lieuDit
        self.doc = nlp(self.name)
        self.potentialLemma = None
        
        if not self.lieuDit:
            self.setLemmaBoundaries()

        self.setDataFrame()
        
        if self.lieuDit:
            self.searchForPotentialLemma()
            
        else:
            lemmaStart, lemmaEnd, preLemma, postLemma = self.lemma_boundaries
            if preLemma >= 0:
                self.analysePreLemma()
            else:
                self.prelemma_df = None
            if postLemma < len(self.doc):
                self.analysePostLemma()
            else:
                self.specific_df = None
                self.postlemma_df = None
                
                
    def setLemmaBoundaries(self):
        # calculate position of generic in the tokens
        lemma_idx = [t.text for t in self.doc].index('Fffff')
        tailLen = (len(self.doc)-1-lemma_idx)
        pre_lemma_idx = lemma_idx-1
        
        # recreate the original for better nlp analysis
        original = ' '.join([t.text for t in self.doc][:lemma_idx])+' '+self.lemma+' '+' '.join([t.text for t in self.doc][lemma_idx+1:])
        self.original = original.strip()

        # reanalyse nlp original and set boundaries of the lemma
        self.doc = nlp(self.original)
        post_lemma_idx = (len(self.doc)-tailLen)
        self.lemma_boundaries=(lemma_idx,post_lemma_idx-1,pre_lemma_idx,post_lemma_idx)
        
    def setDataFrame(self):
        tokens=[]
        # analyze each token in text and set gender an dnumber if exist
        for token in self.doc:
            #print(token.text, token.pos_, token.tag_, token.lemma_)
            tag = token.tag_.split('__')[1]
            gender=None
            number=None
            if len(tag)>2:
                properties = tag.split('|')
                for p in properties:
                    key,value = p.split('=')
                    if 'Gender' == key:
                        gender = value.lower()
                    if 'Number' == key:
                        number = value.lower()
            tokens.append([token.text,token.pos_,gender,number,token.lemma_])
        # return pandas dataframe
        self.df = pd.DataFrame(data=tokens, columns=['text','position','gender','number','lemma'])
        
    def searchForPotentialLemma(self):
        # potential lemma has two conditions: (1) the "...(de...)?" shape or (2) noun exist in fr wiktionary
        # create a pattern for a potential lemma
        pattern = re.compile(r"(NOUN|PRON)+(ADJ)?((DET|ADP)+([A-Z]+)+)?")
        tokenPos = ''.join(self.df['position'].tolist())
        tokenLen = [len(i) for i in self.df['position'].tolist() ]
        match = pattern.search(tokenPos)
        # if we find the pattern we extract the corresponding lemma
        
        if match is not None:
            position = match.start()
            #print('found potential lemma at position',position,'in',tokenPos)
            for i in range(0,len(tokenLen)):
                if sum(tokenLen[:i])== position:
                    self.lemma = self.df['text'].iloc[i]
                    self.lemmatized = self.df['lemma'].iloc[i]
                    #print('lemma is at position',i,'in dataFrame & lemma is',self.lemma)
            
            # if potential lemma is found, query wiktionary to check if noun exists and get origin
            #try:
            #    url = 'http://fr.wiktionary.org/w/api.php?action=parse&page='+self.lemma+'&prop=sections&format=json'
            #    r = requests.get(url)
            #    page = r.json()
            #    langInPage = {}
            #    langHasNoun = {}
            #    for section in page['parse']['sections']:
            #        if section['toclevel']==1:
            #            langInPage[section['number']]=section['anchor']
            #            langHasNoun[section['number']]=False
            #        if section['toclevel']==2 and 'Nom' in section['anchor']:
            #            langHasNoun[section['number'].split('.')[0]]=True
            #    languages = ([langInPage[i] for i in langInPage.keys() if i in langHasNoun.keys()])
            #except:
            #    languages = None
            try:
            	languages = nouns[self.lemma]
            except KeyError:
            	languages = None
            self.potentialLemma = (self.lemma, languages, self.lemmatized)
    
    def analysePreLemma(self):
        lemmaStart, lemmaEnd, preLemma, postLemma = self.lemma_boundaries
        
        # if the token just before the generic is a noun it might as well a adjective 
        if self.df['position'].loc[preLemma] == 'NOUN':
            t, p, g, n, l = self.df.loc[preLemma]
            if l in adj_high_priority:
                self.df.at[preLemma] = [t,'ADJ',g,n,l]
        self.prelemma_df = self.df.loc[:preLemma]#.to_json(orient='records')self.prelemma_df = self.prelemma_df[self.prelemma_d.position != 'PUNCT']
        self.prelemma_df = self.prelemma_df[self.prelemma_df.position != 'PUNCT']
        #print('[INFO] pre lemma',self.prelemma_df)
        
    def analysePostLemma(self):
        lemmaStart, lemmaEnd, preLemma, postLemma = self.lemma_boundaries
        # SPECIAL if the first position after lemma is an ADJ there might be a mistake
        i = postLemma
        pos = self.df['position'].loc[i]
        dfLen = len(self.df)
        #print(self.df['position'])
        while pos in ['PUNCT','DET','AUX','ADJ','CCONJ','NOUN','VERB','PROPN','ADP']:
            
            if pos not in ['ADJ','DET','AUX','NOUN','VERB','PROPN','ADP']:
                # verify 
                if i+1 < dfLen:
                    i += 1
                    pos = self.df['position'].loc[i]
                    continue
                else:
                    break
                    
            t, p, g, n, l = self.df.loc[i]
            
            if t in ['d','du','des']:
                if pos != 'DET':
                    self.df.at[i] = [t,'DET',g,n,'un']
                    break
                else:
                    break
                
            #print(pos,t, p, g, n, l )
            if pos == 'ADJ':        
                # 1 - If adjective is mistaken for a firstname (more common in street name): reset adjective
                if t in firstnames.keys() and i+1 < dfLen :
                    self.df.at[i] = [t,'NOUN',firstnames[t],'sing',t]
                    break
                    
                # 2 - If adjective is mistaken for a grade (more common in street name): reset adjective
                elif l in grades and i+1 < dfLen:
                    self.df.at[i] = [t,'NOUN',g,n,l]
                    break
                    
                # 3 - If Adjective lemma is not in the list of french adjectives top priority, it is unlikely it is one. 
                elif l not in adj_high_priority:
                    self.df.at[i] = [t,'NOUN',g,n,l]
                    break
                    
                else: 
                    if i+1 < dfLen:
                        i += 1
                        pos = self.df['position'].loc[i]
                    else:
                        break
                        
            elif pos in ['NOUN','VERB', 'AUX','PROPN','DET','ADP']:
                # if NOUN or VERB or AUX or DET : CAN BE AN ADJ IF ADJ TOP PRIORITY
                
                if l in adj_high_priority:
                    self.df.at[i] = [t,'ADJ',g,n,l]
                    
                elif l in adj_low_priority and i+1 == dfLen and self.df['position'].loc[i-1] != 'DET' :
                    self.df.at[i] = [t,'ADJ',g,n,l]
                    i += 1
                    break
                    
                elif t in firstnames.keys() and pos != 'NOUN' and i+1 < dfLen:
                    self.df.at[i] = [t,'NOUN',firstnames[t],'sing',t]
                    #print('found first name not in ADJ',t, i)
                    break
                elif t in firstnames.keys() and pos == 'NOUN':
                    break
            if i+1 < dfLen:
                i += 1
                pos = self.df['position'].loc[i]
            else:
                break
            
        specific_df = self.df.loc[i:]
        if not specific_df.empty :
            specific_filtered_list = specific_df.loc[~specific_df['position'].isin(['DET'])].index.tolist()
            # if it is an empty list, there is no specific, we don't filter as it is unlikely that a street ends with a DET POS
            if len(specific_filtered_list) == 0:
                filtered_i = i
            else:
                filtered_i = specific_filtered_list[0]
            #print(specific)
            self.specific_df = self.df.loc[filtered_i:]
            #print(self.specific_df)
            self.postlemma_df = self.df.loc[postLemma:filtered_i-1]
            #print(self.postlemma_df)
            self.specific_df = self.specific_df[self.specific_df.position != 'PUNCT']
            self.postlemma_df = self.postlemma_df[self.postlemma_df.position != 'PUNCT']
        else:
            self.specific_df = self.df.loc[i:]
            self.postlemma_df = self.df.loc[postLemma:i]
        
            self.specific_df = self.specific_df[self.specific_df.position != 'PUNCT']
            self.postlemma_df = self.postlemma_df[self.postlemma_df.position != 'PUNCT']
        
        #print('[INFO] post lemma', self.postlemma_df)
        #print('[INFO] specific', self.specific_df)
    
    def getDataFrames(self):
        if self.lieuDit:
            return self.df, self.potentialLemma
        else:
            return self.prelemma_df, self.postlemma_df, self.specific_df



types = {'nom_voie':str,'numero':np.int32,'code_insee':str,'code_post':str,'nom_ld':str,'x':np.float64,'y':np.float64,'lon':np.float64,'lat':np.float64,'nom_commune':str} 
csvs = set()
hdfs = set()
names = []
for root, dirs, files in os.walk('/Users/fabien/Downloads/BAN_licence_gratuite_repartage/'):
    for name in files:
        if name.endswith('csv'):
            if name.split('_')[-1][:2] == '97':
                csvs.add(name.split('_')[-1][:3])
            else:
                csvs.add(name.split('_')[-1][:2])

        elif name.endswith('h5'):
            if name.split('_')[-1][:2] == '97':
                hdfs.add(name.split('_')[-1][:3])
            else:
                hdfs.add(name.split('_')[-1][:2])
    remaining_csvs = csvs-hdfs
    
for rc in tqdm(list(remaining_csvs),desc='departements'):
    name = 'BAN_licence_gratuite_repartage_' + rc + '.csv'
    root = '/Users/fabien/Downloads/BAN_licence_gratuite_repartage/'
    bandf = pd.read_csv(root+name,sep=';', usecols=list(types.keys()),dtype=types)
        

    #print('[READ CSV]',bandf['numero'].size,'records')
    #print('[READ CSV] line 0:',bandf.iloc[0])

     # copy all nom_ld in Nomvoie si nom_voie est vide et nom_ld ne l'es pas
    bandf['nom_voie'] = bandf.apply(
        lambda row: str(row['nom_ld']).lower() if row['nom_voie'] is np.NAN else row['nom_voie'],
        axis=1
    )

    #print('[READ CSV] processing',len(list(bandf.groupby(['nom_commune','nom_voie']))),'odonyms now')

    potentials= []
    ban = []
    for (city,street), group in tqdm(bandf.groupby(['nom_commune','nom_voie']),desc='street analysis'):
        
        numbers = group[['numero','x','y','lon','lat']].to_json(orient='records')
        record = {"nom_commune":city,"code_insee":group['code_insee'].iloc[0],"code_post":group['code_post'].iloc[0],"nom_voie":street, 'numero':numbers}
        lemma_row = bestCandidate(street)
        
        if lemma_row is not None:
            #print(city,', ',street,', ',lemma_row['lemma'])
            street_prepared = lemma_row['regex'].sub('Fffff', street,1)
            #print(city,', ',street,', ',lemma_row['lemma'])
            if 'Fffff\'' in street_prepared:
                street_prepared = street_prepared.replace('Fffff\'','Fffff ')
            #print(city,', ',street,', ',lemma_row['lemma'])
            try:
                odonym = Odonym(street_prepared,lemma_row['lemma'],False)
            except ValueError as ve:
                print(city, street, street_prepared, ve)
                continue
            except IndexError as ie:
                print(city, street, street_prepared, ie)
                continue
            prelemma, postlemma, specific = odonym.getDataFrames()
            record['prelemma'] = prelemma.to_dict(orient='records') if prelemma is not None else None
            record['postlemma'] = postlemma.to_dict(orient='records') if postlemma is not None else None
            record['specific'] = specific.to_dict(orient='records') if specific is not None else None
            record.update(lemma_row.to_dict())
        else:
            if street.startswith('ldt'):
                street = street[4:]
            elif street.startswith('lieu dit'):
                street = street[8:]
            try:
                odonym = Odonym(street,None,True)
            except ValueError as ve:
                print(city, street, ve)
                continue
            except IndexError as ie:
                print(city, street, ie)
                continue
            specific, potential = odonym.getDataFrames()
            lieudit = {'raw_regex':'','lemma':'lieu dit','signifier_lvl_0':'lieu dit','signifier_lvl_1':'lieu dit','signifier_lvl_2':'voie communication','signifier_lvl_3':None,'signifier_common':'oui','etymology_language':'fr','etymology_region':'France'}

            if potential is not None:
                generic, languages, lemma = potential
                if generic != 'ldt' and generic !='lieu dit':
                    potentials.append({'generic':generic,'languages':languages,'lemma':lemma, 'street':street})
                
            record['prelemma'] = None
            record['postlemma'] = None
            record['specific'] = specific.to_dict(orient='records') if specific is not None else None
            record.update(lieudit)
        ban.append(record)
    bandf2 = pd.DataFrame.from_records(ban)
    
    name = 'store_' + rc + '.h5'
    root = '/Users/fabien/Downloads/BAN_licence_gratuite_repartage/'
    store = pd.HDFStore(root+name,'w')
    store['bandf'] = bandf2
    store.close()

    name = 'zz_potentials_' + rc + '.csv.old'
    root = '/Users/fabien/Downloads/BAN_licence_gratuite_repartage/'
    bandf2potentials = pd.DataFrame.from_records(potentials)
    bandf2potentials.to_csv(root+name)





















