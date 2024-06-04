# PDF_ChatBot

A streamlit chat bot web app that can take files PDFs, embed them into a vector database then using RAG (Retrieval Augmented Generation), can serve back relevant data when queried by a user.

Using Ollama, with the model Llama3, it serves as a private local chat bot to assist users with finding out information about their PDF documents.

To use, install the dependencies from the `requirements.txt` into a python virtual environment. Then run the command `streamlit run PDF_ChatBot.py`.

Disclaimer: the LLM isn't always consistent, when querying, don't ask it super complex questions. Keep it as simple as possible. The query also takes a while to process if you are only running the application using a CPU only.

Your also need to install Ollama and then get Llama3 itself. Once you have Ollama installed simply run `ollama pull llama3` to download Llama3 for your machine.
