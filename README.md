# AI_Warriors

# 🧠 GenAI-Powered Document Processing App

This is a multimodal, secure, SaaS-based RAG (Retrieval-Augmented Generation) solution for intelligent document processing using **Azure OpenAI** or **Gemini** models. It supports **summarization**, **named entity recognition**, **table extraction**, **image understanding**, and **data export in multiple formats** (JSON, XML, PDF, Word, Excel).

---

## 📌 Problem Statement

Manual document processing is tedious, error-prone, and time-consuming. Enterprises deal with a vast amount of unstructured data across invoices, contracts, onboarding forms, and scanned images, often requiring human intervention.

---

## ✅ Solution Highlights

- Upload and process PDFs, Word files, scanned images, and documents with tables/charts.
- Choose between **Gemini** or **Azure OpenAI** models (GPT-3.5, GPT-4, GPT-4o).
- Extract text, tables, images, and metadata from documents.
- Generate summaries (abstract, extractive, or comprehensive).
- Perform Named Entity Recognition (NER).
- Download the processed output in **JSON**, **XML**, **PDF**, **Excel**, and **Word** formats.
- Compliant with security and privacy guardrails (GDPR/SOC2).
- Streamlit-based modern UI with logo branding.

---

## 🎯 Features

| Feature | Description |
|--------|-------------|
| 🔍 Summarization | Abstract, Extractive, or Full Document |
| 🧾 Table Extraction | Structured table data from documents |
| 🧠 Named Entity Extraction | Persons, Locations, Dates, etc. |
| 🖼️ Image Understanding | GPT-4o/GPT-4 native image processing |
| 💾 Multi-format Export | JSON, XML, PDF, Word, Excel |
| 🔐 Security Guardrails | API-based auth, GDPR-compliant pipeline |
| 🔄 Model Selection | Azure OpenAI / Gemini integration |

---

## ⚙️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io)
- **LLM Backend**: Azure OpenAI (GPT-3.5, GPT-4, GPT-4o), Gemini Pro
- **Document Parsing**: LangChain, Unstructured.io, PyMuPDF, Docx, Pillow
- **Data Processing**: Pandas, Python
- **Export**: Python-docx, xlsxwriter, reportlab
- **Security**: API Key-based input, GDPR guardrails

---

## 📂 Directory Structure
📁 your-project/ ├── app.py # Streamlit main application ├── utils/ │ ├── summarization.py # Summarization logic │ ├── extraction.py # Extraction (text, image, table) │ ├── format_exporter.py # Export to Word, PDF, JSON, etc. ├── assets/ │ └── tech.png # Company logo ├── requirements.txt # Python dependencies └── README.md

Install dependencies:
  - pip install -r requirements.txt

Run the app:
  - streamlit run app.py

