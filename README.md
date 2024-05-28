# CSV_ChatBot

A streamlit chat bot web app that can take files (CSVs or PDFs), embed them into a vector database then using RAG (Retrieval Augmented Generation), can serve back relevant data when queried by a user.

Using Ollama, with the model Llama3, it serves as a private local chat bot to assist users with finding out information about their documents.

To use, install the dependencies from the `requirements.txt` into a python virtual environment. Then run the command `streamlit run [.py file of choice]`. Currently there are two files: one for PDFs and one for CSVs.

Disclaimer: the LLM isn't very consistent, when querying, don't ask it super complex questions. Keep it as simple as possible. The query also takes a while to process if you are only running the application using a CPU only.
