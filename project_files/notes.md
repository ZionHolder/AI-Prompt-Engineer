# Course – 1 Intro to RAG

Mini-project, notes, and labs for this course.

## Deliverables
- [x] Lab code
- [x] Project demo
- [x] Notes summary

## 🧠 Key Concepts & Notes

### Module 1 – Foundations of GenAI & Prompt Engineering

## 🧠 Lesson: Introduction to Generative AI

- Two main types of AI: **Discriminative** (classify, analyze) vs **Generative** (create, generate)
- **Discriminative AI** = tells if something is A or B (e.g., puppy or fried chicken)
- **Generative AI** = creates new content (text, images, audio, code)
- **Foundation Models** = broad, general-purpose models trained on massive data (e.g., LLMs like GPT, Gemini)
- **LLMs** = a type of Foundation Model focused on generating human-like language

### Reflection
- Tools like ChatGPT and Gemini are **hybrid systems**—they use **generative models** as the core, but can handle discriminative tasks via fine-tuning or plugins.

## 🧠 Lesson: What are Generative AI Models?

- **LLMs** = (Large Language Models) An application of a **Foundation Model**
- **Foundation Model** = A type of general-use AI that can be used to perform very specific applications due it being trained on large amounts of unstructured data.
- **Foundation Models** are also Generative AI
- **Tuning** = When you label your data to better suit your application
- When using lowly labelled or unlabelled data, prompting can be used (prompt engineering applicable here)
- **FM ADVs** = Performance and Productivity (both due to large traita-set)
- **FM dAVDs** = Costly and not *as* Trustworthy
- **FM Applications** = Not just LLMs. Code, Vision, Chemistry, Climate Change.

## Reflection
- There's a lot of hype around LLMs recently. All the other applications of FMs seem to be overlooked but I can understand why. I feel as well that LLMs are where the money is so that can be a major contributor as well.

## 🧠 Lesson: What is NLP (Natural Language Processing)?

- **NLP** = When you ask/train a computer to understand and comprehend human language
- **Unsctructured Speech** = How computers interpret how we speak or how we speak naturally.
- **Structured Speech** = How computers understand what we say in a form using tags (<>) for example.
- NLP is the bridge or translator between converting from unstructured speech to structured speech and vice versa
- **NLU** = Natural Language Undersatnding - Unstructured Speech to Structured Speech
- **NLG** = Natural Language Generation - Structured Speech to Unstructured Speech
- **NLP Uses** = Machine Translation, Virtual Assistants/Chatbots, Sentiment Analysis, Spam Detection
- NLP is not a single algorithm but a bag of tools used to "solve" the use cases
- **SOME tools of NLP** = First we need input from text or speech-to-text then **Tokenization** (break the text up into smaller parts basically) then **Stemming** (finding the root word of a token) or **Lemmatization** (finding the root meaning of a token), **Part of Speech Tagging** (finding the context of a token) then **N.E.R (Named Entity Recognition)** (giving a token meaning e.g. Trinidad & Tobago - entity: Caribbean Country vs Zion - entity: name of person)

## Relfection
- Cools cools. Before this I was a bit shaky about what NLP is but I believe I have a decent grasp on it now and how it plays a part in LLMs. There of course needs to be an NLP step to do NLU then NGU after the job is done.

## 🧠 Lesson: Intro to In-Context Learning
- **In-context Learning** = Providng examples of what you want to achieve at the same time as giving the instructions.
- **Prompt** = Input given to a gernrative AI to achieve an output. Broken down into two main parts: Instruction and Context
- **Prompt Engineering** = Clarifying a prompt to achieve better output from the LLM.
- A well structured prompt should provide Instruction (What to do) -> Context -> Input data (Data the LLM will be performing the instruction with) -> Output (Where you want the LLM to include the output)

## Reflection
It seems as if majority of people (including myself) are prompting haphazrdly. Unless LLMs like ChatGPT format the input in a certain way for you, I believe that prompting properly even in ChatGPT, Gemini, Llama, Sora, Veo3 etc. can provide more accurate results. As a result, enhancing productivity compared to normal everyday prompting. Of course though, normal everyday prompting has it's place for simple questions or tasks.

## 🧠 Lesson: Intro to Langchain
- **Langchain** = Python Framework that allows devs to easily intergrate LLMs into their
applications
- It is modular and helps put together the process like building blocks or links the parts of the process by chains. It can easily integrate with vector DBs allowing for quick information retrival.
- Langchain can be used with external libraries and models.

## Reflection
I knew a bit about langhchain before so no surprises here. It was cool to learn why there's "Chain" in the name.

## 🧠 Lesson: Advanced Methods of Prompt Engineering

- **Zero-shot Prompting** = Prompting with no examples. The LLM is given a task with no examples
- **One-shot Prompting** = Prompting with one example. The LLM is given a single example of expected output and is expected to do the task modelled like the example.
- **Few-shot Prompting** = Promting with a few examples to give more context. The LLM is given multiple examples and generalises how it should output based on the given output.
- **Chain-of-Thought Prompting** = A prompting method where an LLM is prompted to solve a problem and show the "thought process"eg. breaking down math problems step by step.
- **Self Consistency** = Prompt technique used to generate multiple independent answers to the same question, then evaluates the answers to determine the most consistent result. This increases reliability.
- **Prompt Template** = A base for a prompt that helps with generating effective prompts. Uses placeholders.. kind of like a function.

CODE SNIPPET:
---
from langchain_core.prompts import PromptTemplate

prompt_template = PromptTemplate.from_template(
  "Tell me a {adjective} joke about {content}."
)

prompt_template.format(adjective="funny", content="chickens")

-> "Tell me a funny joke about chickens."
---

- **Agent** = Uses LLMs and Frameworks to perform complex tasks across various domains using different prompts.

## Reflection
I'm a bit tired today so I'm not grasping stuff as well. However, it's cool to learn about different prompting methods. It's giving me some ideas about different things like using Chain-of-Thought prompting to not only do math or coding problems but to do real thinking about day to day problems.

## 🧠 Lesson: LangChain LCEL Chaining Method
- **LCEL** = (LangChain Expression Language) Pattern for building LangChain applications using the pipe (|) operator to connect components.
-  **Runnables** = Connects input and output steps. Two types **Sequential** and **Parallel**
-  **Sequential Runnable** = Runs one after the next using output from 1st Runnable as input to second Runnable
-  **Parallel Runnable** = Uses same input. Processes everything at the same time.

### Module 2 – LangChain Core Concepts

## 🧠 Lesson: LangChain Core Concepts
- **LangChain** = Python framework used to assist in ecooperating LLMs into applications. It consists of parts like agents, output parsers and chat models.
- **PromptTemplate** = Used to translate user's questions or messages into clear instructions. There are types of prompt templatess such as SingleString Prompt Template, Few Show Prompt Template and ChatPrompt Template
- **Ouptut Parser** = Used to help format the output into different formats such as csvs, json, panda data frames

## Reflection
The videos are a little crammed with info but I believe it was an overview. Nothing too new really came up. I will solidify as I proceed.

## 🧠 Lesson: LangChain Chains and Agents for Building Applications
- **Chain** = A tool in LangChain that takes input and passes out input. A chain should comprise of multiple smaller chains that utilises each other's input and output.
- To make a SequentialChain, the logic is basically, making a prompt template, initialising the llm with input and output keys and repeating that with the output key from the previous llm as the input key to the new llm. Then output parsers at the end of each to keep format.
- **Agent** = LLM used to execute tasks. It does not execute tasks on it's own but it uses APIs to do so. Imagine it like an overseer. You ask it what to do and using the APIs, I guess if after processing Human Language, it determines the next best step from the Human Language.

## Reflection
Agents are so cool. Can't wait to start using and building my own processes.

### Module 3 – Flask App with LangChain

---

## 🧪 Labs / Mini-Project

## 🧪 Lab 1 – [Master Prompt Engineering and LangChain PromptTemplates]

### 🛠️ What I Built:
- The lab covered zero-shot, one-shot, few-shot, self-consistency and chain-of-thought prompt engineering techniques. It also introduced basic LangChain prompt template techniques and LECL (LangChain Expression Langue) pipelines.

### 🔑 Key Concepts Practiced:
- PromptTemplate: Used with LangChain to input different variables into a prompt without having to recreate the prompt everytime.
- LLMChain: links prompt → LLM → output
- StructuredOutputParser(): ensures that only python text is displayed.
- RunnableLambda(): Makes python functions compatable with LangChain

### 💡 What Broke / What Was Confusing:
- I had to fix how they were initialising their LLM in the lab. Every initialisation was using default params even when params were passed resulting in unexpected behaviour.
- It was confusing why the model used would give itself more scenarios to repsond to even after the list of "reviews" were exahusted. It made up it's own reviews to respond to depending on the max_new_tokens given. The more tokens, the more the hallucincation, the less tokens, the more accurate the response seemed. At least with model:meta-llama/llama-3-3-70b-instruct on the WatsonxLLM.

### 💥 Learnings:
- Can now generate structured responses by chaining prompts and models
- I now prompt using 5 differnt prompting techniques

## 🧪 Lab 2 – [Build Smarter AI Apps: Empower LLMs with LangChain]

### 🛠️ What I Built:
- I built prompt templates, an AI agent, Sequential Chains using the traditional way but also using langchain LCEL.
- I built a simple chatboy with memory incorating document splitting, retrieval vetors via embeddings, document loaders etc.

### 🔑 Key Concepts Practiced:
- Setting up Chat models using LangChain built-in message types and testing the difrerce between outputs of two models with different temperatures.
- Using PrompTemplates

### 💡 What Broke / What Was Confusing:
- Using MessagesPlaceholder wasn't.. confusing but I'm not solid with it.
- Text Splitters.. I know how they work but it's a little confusing sometimes about how they split up the tokens and how the chunks are made. Some text splitters are easy to understand, others it can be a little confusing.
- LLMs (at least the models in this certification.. hallucinate and add more scenarios to make up max token limits)

### 💥 Learnings:
- Different models are better suited for different things
- I learnt about SystemMessages setting the role of the LLM.
- There are different useful DocumentLoaders for different things in the LangChain framework i.e. PyPDFLoader and WebBaseLoader
- Embedding models and retrievers (ParentDocument retirever uses large chunks and small chunks.. similar to how cache and main maemory works large chunks -> smaller chunks)
- Making a RetrievalQA chain
- Adding memory to conversations
- Using RunnablePassthrough for LCEL Sequential chains
- Target specific parts of the chain's out put to get the text only e.g. response['this'] or response['that']
- for custom tools you can use the @tool decorator or the Tool class directly
(imports: from langchain_core.tools import Tool
from langchain.tools import tool)
- Toolkit: Collection of tools eg using a list.
- Use from langchain.agents import create_react_agent, AgentExecutor. create_react_agent to create the agent and AgentExecutor to manage it. The format for the agent executer must have the tools in it i.e. {tools} && {tool_names} but also: Thought, Action, Action input, Observation then Thought and Final answer. This is the ReAct framework.
