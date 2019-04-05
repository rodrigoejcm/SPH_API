#! pip install num2words

import re
import contractions_en
import sys
import html
from nltk.corpus import stopwords
from num2words import num2words



#### LOWER

def ct_lower(text):
  return text.lower()

#### EXPADIR CONTRACOES DO INGLES
### melhorar pra dectar quando esta entre duas letras pra remover so esses
### pode ser melhorado com https://pypi.org/project/pycontractions/

def ct_expand_contractions_english(text):
  #REMOVE ' e ’
  text = re.sub(r"'", "", text)
  text = re.sub(r"’", "", text)
  for key, value in contractions_en.contractions.items():
    text = re.sub(r""+key+"", value, text)
  return text

#### REMOVE >> Twitter user

def ct_remove_user_names(text):
  return re.sub( r'(^|[^@\w])@(\w{1,30})\b', ' ', text )

#### REMOVE >> Twitter hashtag 

def ct_remove_hashtags(text):
  return re.sub( r'(^|[^@\w])#(\w{1,30})\b', ' ', text )

#### REMOVE >>  urls

def ct_remove_urls(text):
  text = re.sub(r"http\S+", "", text)
  text = re.sub(r"https\S+", "", text)
  text = re.sub(r"www\S+", "", text)
  return text

#### Convert Entidades Caracter
def ct_convert_character_entities(text):
  return html.unescape(text)

#### REMOVE >>  Símbolos

def ct_remove_symbols(text,punctuatio):
  text = html.unescape(text)

  if punctuatio:
    return re.sub(r"[-()\"#/@;:<>{\[\]}`+=~|.!?,]", " ", text)
  else:
    return re.sub(r"[-()\"#/@<>{\[\]}`+=~|]", " ", text)


#### CONVERTE números para palavras
def ct_convert_number_to_words(text):
  numbers_re = re.compile(r'[0-9]+')
  text = numbers_re.sub(lambda m: num2words(int(m.group()),lang='en'), text)
  return text

### REMOVE >>  Non ASCII
#### emojis quebra de libra e etc

def ct_remove_non_ascii(text):
  return re.sub(r"[^\x1F-\x7F]+", " ", text)

### REMOVE >>  EXCESSIVE SPACES
def ct_remove_excessive_space(text):
  return " ".join(text.split())

#### CONVERTE alguns símbolos em palavras
def ct_symbols_for_word(text):
  text = re.sub(r"%", " per cent ", text)
  text = re.sub(r"\$", " dolars ", text)
  text = re.sub(r"&", " and ", text)
  return text

#### Remove Stop words
def ct_remove_stop_words(text):
  stops = set(stopwords.words("english"))
  text = text.split()
  text = [w for w in text if not w in stops and len(w) >= 3]
  text = " ".join(text)
  return text


#### APlica todos os métodos

def clean_text_full(text, expand=True, punctuation=True, stopwords=True):


  text = ct_lower(text)
  
  if expand:
    text = ct_expand_contractions_english(text)
  
  text = ct_remove_user_names(text)
  
  text = ct_remove_hashtags(text)
  
  text = ct_remove_urls(text)
  
  text = ct_convert_number_to_words(text)
  
  text = ct_convert_character_entities(text)
  
  text = ct_symbols_for_word(text)
  
  text = ct_remove_non_ascii(text)
  
  text = ct_remove_symbols(text,punctuation)
  
  text = ct_remove_excessive_space(text)

  if stopwords:
    text = ct_remove_stop_words(text)	
	
  if text == "":
    text = "EMPTY STRING"
  
  return text
