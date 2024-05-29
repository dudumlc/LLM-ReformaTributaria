# BIBLIOTECAS USADAS
import pandas as pd
import numpy as np

import os 

from dotenv import load_dotenv, find_dotenv

from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

import streamlit as st
import pandas as pd
from scipy.spatial.distance import cosine

load_dotenv(find_dotenv())

st.secrets()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# DATAFRAME COM OS TEXTOS VETORIZADOS
df=pd.read_csv('embeddings_url.csv', index_col=0)
df['ada_embedding'] = df['ada_embedding'].apply(eval).apply(np.array)
df = df.reset_index()
df = df.drop(59, axis=0)

# MODELO DE EMBEDDING 
embedding = OpenAIEmbeddings(api_key= OPENAI_API_KEY, model='text-embedding-ada-002')

# MODELO GENERATIVO
llm = OpenAI( api_key = OPENAI_API_KEY, 
              temperature = 0.2,
              max_tokens=200,
              model='davinci-002'
             )

# CABEÇALHO DA INTERFACE
st.header("🤖 ASSITENTE TRIBUTARISTA - ESPECIALISTA EM REFORMA TRIBUTÁRIA")

# ENVIO PERGUNTA DO USUÁRIO
text = st.text_input("Faça sua pergunta",)

envio_pergunta = st.button('Enviar')

# PERGUNTA VETORIZADA
texto_vetorizado = embedding.embed_query(text= text)

# CALCULANDO A DISTANCIA DOS TEXTOS DOS SITES COM A PERGUNTA
df["distances"] = df["ada_embedding"].apply(lambda x: cosine(texto_vetorizado, x))

# PEGANDO OS TRÊS TEXTOS MAIS SIMILARES A PERGUNTA
contexto_top_tres = []

for i,v in enumerate(df.sort_values(by='distances', ascending=True).head(3)['Texto']):
    contexto_top_tres.append(v)
    contexto_top_tres.append('----------')

contexto_prompt = '\n'.join(contexto_top_tres)

# PEGANDO OS TRÊS TEXTOS MAIS SIMILARES A PERGUNTA
urls = []

for i,v in enumerate(df.sort_values(by='distances', ascending=True).head(3)['URL']):
    urls.append(v)

# TEMPLATE DO PROMPT A SER ENVIADO AO MODELO

PROMPT_TEMPLATE_TEXT = """
                          Atue como um especialista sore reforma tributária do sistema tributário brasileiro que trabalha respondendo dúvidas sobre o assunto. 
                          Reponda, de maneira profissional e completa, a questão baseada apenas no seguinte contexto: 
                          {context}
                          Responda a questão abaixo de acordo com o contexto axima: 
                          {pergunta}.
                          Não diga em sua resposta que você está se baseando no contexto. Caso a pergunta não esteja relacionada ao contexto, diga que você não é permitido responder ou que não tem conhecimento.
                       """

# CONFIGURANDO O TEMPLATE DO PROMPT
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE_TEXT)

# ASSOCIANDO ÀS VARIÁVEIS DO PROMPT TEMPLATE
prompt = prompt_template.format(context=contexto_prompt, pergunta=text)

# CONFIGURANDO O MODELO GENERATIVO DE CHAT
chatllm = ChatOpenAI(api_key= OPENAI_API_KEY,
                     temperature=0.4)

# GERANDO A RESPOSTA DO MODELO
resposta_chatllm = chatllm.invoke(prompt)

if envio_pergunta:

    # PRINTANDO A RESPOSTA DO MODELO
    st.write(resposta_chatllm.content)

    with st.expander(label="Fontes de Dados"):
        st.markdown(f"<h1 style='font-size:15px;'>Referências:</h1>", unsafe_allow_html=True)
        st.markdown(f"1. [{urls[0]}](%s)" % urls[0])
        st.markdown(f"2. [{urls[1]}](%s)" % urls[1])
        st.markdown(f"3. [{urls[2]}](%s)" % urls[2])

