# Using OpenAI and Langchain to answer question about text 

🤖Ask questions about the text in natural language🤖

💪 Built with [LangChain](https://github.com/hwchase17/langchain)

💪 Based on [Notion-QA](https://github.com/hwchase17/notion-qa)

# 🌲 Environment Setup

Set your OpenAI API key (if you don't have one, get one [here](https://platform.openai.com/))

```shell
export OPENAI_API_KEY=....
```

## Ingest text 

```shell
python ingest.py path_to_file.txt
```

## 💬 Ask a question
In order to ask a question, run a command like:

```shell
python qa.py "What is the work from home policy"
```
