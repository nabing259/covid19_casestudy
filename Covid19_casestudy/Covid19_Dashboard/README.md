# Covid-19 Dashboard

A Streamlit-powered interactive dashboard for visualizing and analyzing global COVID-19 data, enhanced with AI-generated insights using Google Gemini.

---

## Table of Contents

- [Covid-19 Dashboard](#covid-19-dashboard)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Features](#features)
  - [Data Sources](#data-sources)
  - [Setup \& Installation](#setup--installation)
  - [Usage](#usage)
  - [Project Structure](#project-structure)
  - [Key Functionalities](#key-functionalities)
  - [AI Integration](#ai-integration)

---

## Project Overview

This dashboard provides a comprehensive, interactive view of COVID-19 cases, recoveries, and deaths worldwide. Users can select countries, date ranges, and case types to explore trends, spikes, and regional breakdowns. The dashboard also features an AI assistant that summarizes the data and answers user questions in plain language.

---

## Features

- **Interactive Visualizations:** Maps, line/bar/area charts, sunburst, and donut charts.
- **Country & Date Selection:** Filter data by country and custom date ranges.
- **Status Selection:** Analyze Confirmed, Recovered, Death, or Overall data.
- **Key Metrics:** Total cases, recoveries, deaths, active cases, rates, spikes, and more.
- **Regional Breakdown:** Province/state-level analysis for selected countries.
- **AI Assistant:** Summarizes data and answers user questions using Google Gemini.
- **Responsive UI:** Built with Streamlit for easy web deployment.

---

## Data Sources

The dashboard uses publicly available datasets:

- [COVID-19 Confirmed Cases](https://raw.githubusercontent.com/nabing259/covid19_casestudy/refs/heads/main/Covid19_casestudy/covid-19-dataset/covid_19_confirmed_v1.csv)
- [COVID-19 Deaths](https://raw.githubusercontent.com/nabing259/covid19_casestudy/refs/heads/main/Covid19_casestudy/covid-19-dataset/covid_19_deaths_v1.csv)
- [COVID-19 Recoveries](https://raw.githubusercontent.com/nabing259/covid19_casestudy/refs/heads/main/Covid19_casestudy/covid-19-dataset/covid_19_recovered_v1.csv)

See [Covid19_Dashboard/data/README.md](Covid19_Dashboard/data/README.md) for more details on data formats and preprocessing.

---

## Setup & Installation

1. **Clone the repository:**

   ```sh
   git clone <repository-url>
   cd Covid19_Dashboard
   ```

2. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```

   Or, if using Poetry:

   ```sh
   poetry install
   ```

3. **Set up Google Gemini API key:**
   - Obtain an API key from Google Gemini.
   - Replace the placeholder in `main.py`:

     ```python
     genai.configure(api_key="YOUR_API_KEY")
     ```

---

## Usage

1. **Run the dashboard:**

   ```sh
   streamlit run Covid19_Dashboard/src/main.py
   ```

2. **Interact with the dashboard:**
   - Use the sidebar to select:
     - Country (or "World" for global view)
     - Date Range
     - Status (Confirmed, Recovered, Death, Overall Analysis)
   - Explore the main dashboard for visualizations and metrics.
   - Switch to the Data AI tab for AI-generated summaries or to ask questions about the data.

---

## Project Structure

```
Covid19_Dashboard/
├── data/
│   └── README.md
├── src/
│   ├── main.py
│   └── utils.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

- **data/**: Documentation about the datasets.
- **src/main.py**: Main Streamlit app.
- **src/utils.py**: Utility functions.
- **requirements.txt**: Python dependencies.
- **pyproject.toml**: Project metadata and dependencies.

---

## Key Functionalities

- **Data Loading & Preprocessing:** Loads CSVs directly from GitHub, cleans, and merges them for analysis.
- **Visualization:** Uses Plotly for interactive charts and maps.
- **Metrics Calculation:** Computes totals, rates, spikes, moving averages, and more.
- **Regional Analysis:** Province/state breakdown for countries with sub-regions.
- **AI Assistant:** Uses Google Gemini to generate summaries and answer user questions.

---

## AI Integration

- **Gemini API:** The dashboard integrates with Google Gemini to provide:
  - Plain-language summaries of the selected data.
  - Answers to user questions based on the current data view.
- **Prompt Engineering:** Prompts are dynamically generated based on user selections and current metrics.

---
