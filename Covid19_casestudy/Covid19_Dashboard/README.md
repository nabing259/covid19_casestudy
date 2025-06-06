# Covid-19 Dashboard (AI powered) (https://covid19casestudy-ehupdbjzqwngbmvjpyw8bu.streamlit.app/)

A Streamlit-powered interactive dashboard for visualising and analysing global COVID-19 data, enhanced with AI-generated insights using Google Gemini.

---
## 📚 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Data Sources](#data-sources)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Key Functionalities](#key-functionalities)
- [AI Integration](#ai-integration)
- [Tech Stack](#tech-stack)

---

## 🧭 Project Overview

This project presents a dynamic COVID-19 Dashboard that allows users to interactively explore the spread and impact of the pandemic across the globe. It displays statistics on confirmed cases, recoveries, and deaths, with the added power of **AI-generated insights** from **Google Gemini**.

Users can interact with charts, filter by countries and date ranges, and get plain-language summaries of the data through an integrated AI assistant.

---

## ✨ Features

- **📈 Interactive Charts:** Line, bar, area, sunburst, donut charts and maps powered by Plotly.
- **🌍 Global & Regional Analysis:** Select specific countries or view global data.
- **📅 Custom Date Range:** Visualise data for any custom period.
- **📊 Case Status Selection:** View Confirmed, Recovered, Deaths, or all combined.
- **📌 Key Metrics:** Automatically computed totals, active cases, death rates, spikes, moving averages.
- **🧠 AI Assistant:** Integrated with Google Gemini for summaries and Q&A.
- **📱 Responsive Interface:** Built using Streamlit for a clean, interactive UI.

---

## 📂 Data Sources

This dashboard uses publicly available and regularly updated datasets from GitHub:

- **Confirmed Cases:**  
  [covid_19_confirmed_v1.csv](https://raw.githubusercontent.com/nabing259/covid19_casestudy/refs/heads/main/Covid19_casestudy/covid-19-dataset/covid_19_confirmed_v1.csv)
- **Deaths:**  
  [covid_19_deaths_v1.csv](https://raw.githubusercontent.com/nabing259/covid19_casestudy/refs/heads/main/Covid19_casestudy/covid-19-dataset/covid_19_deaths_v1.csv)
- **Recoveries:**  
  [covid_19_recovered_v1.csv](https://raw.githubusercontent.com/nabing259/covid19_casestudy/refs/heads/main/Covid19_casestudy/covid-19-dataset/covid_19_recovered_v1.csv)

---

## ⚙️ Setup & Installation

### 1. Clone the Repository

    git clone <repository-url>
    cd Covid19_Dashboard

### 2. Create a Virtual Environment (using `uv`)

    uv venv
    uv pip install -r requirements.txt
### 3. Set Up Gemini API Key

Get your API key from **Google Gemini** and configure it in the code.

  In `main.py`, add the following line:
    
          genai.configure(api_key="YOUR_API_KEY")

    
 ### 4. Usage

  To run the dashboard locally:

    uv run streamlit run src/main.py
  Once launched, the dashboard will open in your default browser.

   #### What You Can Do:
  - Choose a country or view the worldwide data.

  - Select a date range to focus on a particular period.

  - Pick the case status to analyse: Confirmed, Recovered, Deaths, or Overall.

  - View interactive charts and metrics.

  - Switch to the "Data AI" tab to get summaries or ask data-specific questions powered by Gemini.

## 5. Key Functionalities

- 📥 **Data Loading**: Directly loads and merges CSV data from GitHub.

- 🧹 **Data Preprocessing**: Cleans and structures the dataset for analysis.

- 📊 **Visualisation Engine**: Built using Plotly for engaging and responsive visuals.

- 📈 **Metrics Analysis**:
  - Total & Active cases  
  - Death and recovery rates  
  - Weekly moving averages  
  - Spike detection  

- 📌 **Regional Analysis**: Drill down into states/provinces (where data is available).

- 💬 **AI Q&A**: Get simplified answers to data queries using Gemini.

---

## 6. 🤖  AI Integration

**Powered by Google Gemini**

- **Smart Summaries**: Generates text-based summaries for the current data view.

- **Interactive Q&A**: Ask questions like:
  - "Which country had the highest spike in April 2021?"
  - "What’s the recovery rate for Canada in July 2020?"

---

### 7. Prompt Engineering

Dynamic prompts are crafted based on your current selections — country, date range, status type, and metrics — to deliver relevant and accurate AI responses.
## 🧰 Tech Stack

- **Streamlit**: Web framework for Python apps  
- **Plotly**: Interactive visualisations  
- **Pandas**: Data processing  
- **NumPy**: Numerical computing  
- **Google Gemini API**: AI assistant integration  
- **UV**: Fast virtual environment and dependency management  
