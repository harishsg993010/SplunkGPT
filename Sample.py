def runPrompt():
  index = faiss.read_index("training.index")

  with open("faiss.pkl", "rb") as f:
    store = pickle.load(f)

  store.index = index

  #TODO: Add your Master Prompt

  with open("training/facts/master.txt", "r") as f:
    promptTemplate = """*write the splunk SPL Query*
---
ConversationHistory: {history}
---
MemoryContext: {context}
---
Human: {question}
Bot:"""

  prompt = Prompt(template=promptTemplate,
                  input_variables=["history", "context", "question"])

  llmChain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0.25))

  def onMessage(question, history):
    docs = store.similarity_search(question)
    contexts = []
    for i, doc in enumerate(docs):
      contexts.append(f"Context {i}:\n{doc.page_content}")
      answer = llmChain.predict(question=question,
                                context="\n\n".join(contexts),
                                history=history)
    return answer

  history = []
  while True:
    question = input("Ask a question > ")
    answer = onMessage(question, history)
    print(f"InSplunkGPT: {answer}")
    history.append(f"SOC Analyst: {question}")
    history.append(f"InSplunkGPT: {answer}")
