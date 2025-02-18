# RAG-Powered Personal Assistant Chatbot
This is an app where the intention is to answer any personal or professional questions that a person might have about me. As a fun project, I did this to learn more about the engineering that has happened to make LLM's useful. I thought this would be a creative implementation of a RAG (Retrieval-Augmented Generation) system, which is a combination of informational retrieval and prompt engineering to create a customized LLM chat application that is powerful enough to avoid having the need to fine-tune an LLM on my personal information. After testing it, I believe it is working quite well and I will continue to add new information about myself to make it even better!

## Who is this for?
I hope this will be used by **RECRUITERS** or potential future **EMPLOYERS** as the proof needed to show my depth as a potential candidate in terms of my engineering ability and ability to quickly learn new technological advancements.

However, I hope it will also be used by friends or perhaps potential new friends who are interested in learning more about me. This is not a replacement for me as a person, but I hope it will encourage others to get to know more on a deeper level. Whether you are a recruiter or a person in my life, I am much more than what I can include in this application. Just like everyone, I am a person with deep convictions and desires, and I hope that this application will not replace me but enhance others to know me better. And there will always be information that is not included here as only more information will be given through true relationships in time :smile:

## Quick Techincal Background of the App
![RAG Workflow Img](info/rag_workflow_diagram.png)


*RAG Workflow By LangChain in their [Self-Reflective RAG with LangGraph](https://blog.LangChain.dev/agentic-rag-with-langgraph/) article*


### OpenAI Models
* LLM: [gpt-4o-mini](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/)
    * Their latest affordable LLM offered is responsible for taking questions in my app and giving the answers at the end.
* Embedding Model: [text-embedding-3-small](https://openai.com/index/new-embedding-models-and-api-updates/)
    * Their latest smaller embedding model is responsible for indexing documents and then embedding the user questions, which are later used to pull relevant documents to answer the user's question.

### Tools to build the RAG System
This app was entirely built on [**LangChain**](https://www.LangChain.com/), which is an open-source tool that has a lot of great tools to be able to build high-functioning LLM applications. LangChain allowed me to build this complex RAG pipeline or in their words "chain", in a relatively easy way. The challenge for me was to learn LangChain as a tool and more specifically the logic and tools used to build a RAG pipeline.

The Vector Store or the database that would be responsible for storing my documents and later pattern matching between documents and the user's input question was the open-source [**Chroma**](https://www.trychroma.com/) Database tool that allows you to quickly create a local database locally and start using it with open available embedding models like OpenAI's *text-embedding-3-small* model.

These are the learning sources I used to build this RAG System:
1. [*RAG From Scratch*](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x) 14-part course offered by LangChain's YouTube channel, and it offered way more complicated RAG powered approaches not needed in my application, but was a great course!
2. [*Build a Retrieval Augmented Generation (RAG) App*](https://python.LangChain.com/docs/tutorials/rag/) article provided by LangChain that has the same ideas as the YouTube course above.
3. [*Conversational RAG*](https://python.LangChain.com/docs/tutorials/qa_chat_history/) article provided by LangChain that allowed me to include chat history to make the dialogue between the LLM and user seem more seamless.

### How was the front-end app interface developed?
The front-end app was built using [**Streamlit**](https://streamlit.io/), which is a powerful python library that is able to quickly build web applications that integrates well with ML models. Their new additions of adding chat interfaces and chat features allowed the integration of LangChain's output a breeze!

### App Deployment
This app is currently running in an [AWS EC2 instance](https://aws.amazon.com/ec2/), since this is a constant running python script. I used online resources to help me configure the EC2 instance to allow traffic to come in at a certain port that the application is running, how to connect a domain to the server and port, and to have the process always be running in the background.

Resources:
1. [How to host any app on AWS EC2 + custom domain for FREE](https://youtu.be/gyizrcHfkcU?si=qaOKIBqlE3zveE3G): YouTube video by *Thinja*
2. [AWS EC2 running a Python script continuously, EC2 setup with Python tutorial](https://youtu.be/xXirbnUB3NU?si=LUWuzCLdFGOZYz1h): YouTube video by *Tech with Hitch*
3. [Nohup Command in Linux](https://www.digitalocean.com/community/tutorials/nohup-command-in-linux): Article by DigitalOcean that helped me to run the app continuously in EC2.