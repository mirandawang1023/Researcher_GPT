import os
from dotenv import find_dotenv, load_dotenv
import requests
import json
from langchain import OpenAI, LLMChain, PromptTemplate
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
import openai
import streamlit as st

find_dotenv()
load_dotenv(find_dotenv())
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


#1.serp service to get list of relevant articles
def search(query):
    url = "https://google.serper.dev/search"

    payload = json.dumps({
        "q":query
    })
    headers = {
        'X-API-KEY': SERPAPI_API_KEY,
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    response_data = response.json()

    print("search_results:", response_data)
    return response_data

query="Comment on the series White Lotus?"
search(query)

#2.llm to choose the best articles, and return urls scrape from medium or twitter
def find_best_article_urls(response_data,query):
    #turn json into string
    response_str = json.dumps(response_data)
    #create llm to choose best articles
    llm = OpenAI(model_name = 'gpt-3.5-turbo', temperature=.7)
    template = """
    You are a world class journalist & researcher, you are extremely good at find most relevant articles to certain topic;
    {response_str}
    Above is the list of search results for the query{query}.
    Please choose the best 3 articles from the list, return only an array of the urls, do not include anything else; return only an array of the urls, do not include anything
    """

    prompt_template = PromptTemplate(
        input_variables = ["response_str", "query"], template=template)
    
    article_picker_chain = LLMChain(
        llm=llm, prompt=prompt_template, verbose=True)
    
    urls = article_picker_chain.predict(response_str=response_str, query=query)

    #Convert string to list
    url_list= json.loads(urls)
    print(url_list)

    return url_list

result = search(query)
urls = find_best_article_urls(result, query)
print(urls)
    
#3.get content for each articles from urls and make summaries
def get_content_from_urls(urls):
    # use unstructuredURLLoader
    loader = UnstructuredURLLoader(urls = urls)
    data = loader.load()

    return data 

def summarise(data, query):
    text_splitter = CharacterTextSplitter(separator = "\n", chunk_size = 3000, chunk_overlap=200, length_function=len)
    text = text_splitter.split_documents(data)

    llm = OpenAI(model_name = "gpt-3.5-turbo", temperature = .7)
    template = """
    {text}
    You are a world class journalist, and you will try to summarise the text above in order to create a twitter thread about {query}
    Please follow all of the following rules：
    1/ Make sure the content is engaging, informative with good data
    2/ Make sure the content is not too long, it should be no more than 3-5 tweets
    3/The content should address the {query} topic very well
    4/The content needs to be viral, and get at least 1000 likes
    5/The content needs to be written in a way that is easy to read and understand
    6/The content needs to give audience actionable advice & insights too

    SUMMARY:
    """

    prompt_template = PromptTemplate(input_variables = ["text","query"], template = template)

    summarizer_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)

    summaries = []

    for chunk in enumerate(text):
        summary = summarizer_chain.predict(text=chunk, query=query)
        summaries.append(summary)

    print(summaries)
    return summaries

results = search(query)
urls = find_best_article_urls(results, query)
data = get_content_from_urls(urls)
summaries = summarise(data, query)

#4.turn summarization into twitter thread
def generate_thread(summaries, query):
    summaries_str = str(summaries)

    llm = OpenAI(model_name = "gpt-3.5-turbo", temperature=.7)
    template = """
    {summaries_str}
    
    You are a world class journalist & twiiter influencer, text above is some context about {query}
    Please write a viral twitter thread about {query} using the text above, and following all rules below:
    1/ The thread needs to be engaging, informative with good data
    2/ The thread needs to be around than 3-5 tweets
    3/ The thread needs to addresss the {query} topic very well
    4/ The thread needs to be viral, and get at least 1000 likes
    5/ The thread needs to be written in a way that is easy to read and understand
    6/ The thread needs to give audience actionable advice & insights too

    Twitter Thread:
    """

    prompt_template = PromptTemplate(input_variables=["summaries_str", "query"], template=template)
    twitter_thread_chain = LLMChain(llm = llm, prompt = prompt_template, verbose=True)

    twitter_thread = twitter_thread_chain.predict(summaries_str=summaries_str, query = query)

    return twitter_thread

results = search(query)
urls = find_best_article_urls(results, query)
data = get_content_from_urls(urls)
summaries = summarise(data, query)
twitter_thread = generate_thread(summaries, query)
print(twitter_thread)

def main():
    st.set_page_config(page_title = "Generate twitter thread", page_icon=":bird:", layout="wide")

    st.header("Generate twitter thread :bird:")
    query = st.text_input("Topic of twitter thread")

    if query:
        print(query)
        st.write("Generating twitter thread for: ", query)

        search_results = search(query)
        urls = find_best_article_urls(search_results,query)
        data = get_content_from_urls(urls)
        summaries = summarise(data, query)
        thread = generate_thread(summaries, query)

        with st.expander("search results"):
            st.info(search_results)
        with st.expander("best urls"):
            st.info(urls)
        with st.expander("data"):
            st.info(data)
        with st.expander("summaries"):
            st.info(summaries)
        with st.expander("thread"):
            st.info(thread)
if __name__ == '__main__':
    main()
