# 📊 CSV Analyst Agent

> Upload any CSV file and ask questions about it in plain English. An AI agent writes and runs the pandas code for you and replies with the answer.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-Agent-1C3C3C)](https://www.langchain.com/)
[![Groq](https://img.shields.io/badge/LLM-Groq-F55036)](https://groq.com/)

🔗 **Live Demo:** [https://csv-analyst-agent-89.streamlit.app/](#)


---

## 📸 Screenshots

| Upload a CSV and preview it | Ask questions in natural language |
|---|---|
| ![Upload](<img width="752" height="507" alt="image" src="https://github.com/user-attachments/assets/84dd2892-f661-4863-9d18-c12aa20a91df" />
) | ![Chat](<img width="727" height="536" alt="image" src="https://github.com/user-attachments/assets/252cc2b6-c85f-4934-9389-67a99e75d8c4" />
) |

---

## ✨ Features

- **Chat with your data:** ask questions like *"give me the mean salary of each department"* and get a formatted answer.
- **Agent with tools:** the LLM decides which tool to use (inspect, filter, or analyze) to answer each question.
- **Conversation memory:** the agent remembers earlier messages, so you can ask follow-up questions.
- **Instant preview:** the first rows of your file are shown right after upload, along with the row count.
- **Simple Streamlit UI:** no setup beyond uploading a file and typing a question.

---

## 🧠 How It Works

```
CSV upload ──► pandas DataFrame ──► LangChain agent (Groq LLM)
                                        │
                  ┌─────────────────────┼─────────────────────┐
                  ▼                     ▼                     ▼
           get_data_info           filter_toll           analyze_data
        (schema, shape, head)   (pandas .query())    (pandas expressions)
                                        │
                                        ▼
                         Answer shown in the Streamlit chat
```

The agent is given three tools:

| Tool | What it does |
|---|---|
| `get_data_info` | Returns the shape, column names, data types and first 5 rows of the uploaded CSV |
| `filter_toll` | Filters rows using a pandas query condition, e.g. `Salary > 60000 and Department == 'Engineering'` |
| `analyze_data` | Runs a pandas expression on the DataFrame, e.g. `df.groupby('Department')['Salary'].mean()` |

The agent is built with LangChain's `create_agent`, uses an `InMemorySaver` checkpointer for conversation memory, and runs the `openai/gpt-oss-20b` model through the Groq API.

---

## 🛠️ Tech Stack

- **Language:** Python
- **UI:** Streamlit
- **Agent framework:** LangChain + LangGraph
- **LLM provider:** Groq (`langchain-groq`)
- **Data handling:** pandas
- **Deployment:** Streamlit Community Cloud, with code hosted on GitHub

---

## 🚀 Run It Yourself

This app uses the Groq API, so **you need your own free Groq API key** to use it. The code is open, so you can plug in your key and run your own copy.

### 1. Get a Groq API key
Create a free account at [console.groq.com](https://console.groq.com/) and generate an API key.

### 2. Clone the repository
```bash
git clone https://github.com/Ayush-kumar-jha953/CSV-ANALYST-AGENT.git
cd CSV-ANALYST-AGENT
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API key
Create a file named `.env` in the project folder:
```
GROQ_API_KEY=your_groq_api_key_here
```
> ⚠️ Never commit your `.env` file or share your key publicly. It is already listed in `.gitignore`.

### 5. Start the app
```bash
streamlit run app.py
```
Open the local URL shown in your terminal, upload a CSV (a sample file, `indianemployee.csv`, is included in this repo) and start asking questions.

---

## ☁️ Deploy Your Own Copy on Streamlit Cloud

1. Fork this repository to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Click **New app**, select your fork, and set the main file to `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
5. Click **Deploy**. Your own CSV analyst agent will be live in a minute or two.

---

## 💬 Example Questions

- "How many rows and columns does this file have?"
- "Give me the mean salary of each department."
- "Who are the employees in Engineering earning more than 70000?"
- "Which department has the highest total salary?"
- "Who joined most recently?"

---

## 📁 Project Structure

```
CSV-ANALYST-AGENT/
├── app.py               # Agent, tools and Streamlit UI
├── indianemployee.csv   # Sample dataset to try the app
├── requirements.txt     # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚠️ Limitations

- The `analyze_data` tool runs model-generated pandas expressions, so run this app only on data and in environments you trust.
- Conversation memory is kept in the running session only and resets when the app restarts.
- Answer quality depends on the underlying LLM, so verify important numbers.

---

## 🔮 Future Improvements

- Chart generation (bar, line, histogram) from natural language
- Support for Excel files and multiple CSVs
- Safer, sandboxed code execution
- Downloadable results
- Per-user conversation sessions

---

## 👤 Author

**Ayush Kumar Jha**
GitHub: [@Ayush-kumar-jha953](https://github.com/Ayush-kumar-jha953)
LinkedIn: [www.linkedin.com/in/ayush-jha-ds](#)

If you found this project useful, consider giving it a ⭐!
