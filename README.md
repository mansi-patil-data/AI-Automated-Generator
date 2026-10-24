# Automated Insight Generator

A Streamlit business dashboard that loads CSV/Excel exports, normalizes and validates business fields, computes KPIs with Pandas/NumPy, visualizes trends with Plotly, optionally answers questions and generates executive recommendations with Google's Gemini API, and downloads reports.

## Features

- CSV, `.xlsx`, and `.xls` upload with normalized business field names.
- Automatic category-column normalization plus dynamic region, product, category, and date filters.
- Exact duplicate row removal and safe parsing of dates and numeric fields.
- Revenue, expenses, profit, profit margin, and average order value KPIs.
- Date/dimension filtering for dashboard metrics, charts, highlights, AI answers, insights, and reports.
- Best/worst region and product rankings, monthly trends, and revenue MoM growth.
- Interactive monthly, regional, category, and product Plotly charts with tooltips and export.
- Exportable Markdown, CSV, and PDF reports with optional AI recommendations.
- Natural-language questions answered from aggregated dashboard metrics when `GEMINI_API_KEY` is configured.
- Common Ask AI questions and executive insights fall back to deterministic local Pandas analytics when Gemini is missing or unavailable.
- Browser voice input with editable transcripts, plus optional answer playback using SpeechSynthesis.
- Revenue-drop and low-performing category alerts, plus an interactive revenue/expense what-if simulator.
- PDF executive summaries generated with `fpdf2`.
- Sidebar theme customization for accent and card colors, with glass-effect cards and dynamically themed Plotly charts.
- Sidebar light/dark mode, side-by-side region/product comparisons, and PNG/SVG Plotly modebar export.
- Optional structured recommendations through Google's current `google-genai` Python SDK.
- Markdown report export; AI access is not required to use the dashboard or report.

Recognized columns include date, region, product, revenue, expenses, profit, and sales quantity. Common alternatives such as `Order Date`, `Sales Region`, `Net Sales`, `Cost`, and `Units Sold` are mapped to canonical names. At least one recognized metric is required.

### Metric conventions

- Rows with usable revenue count as orders for average order value; the input schema has no order identifier.
- If usable profit values exist, totals use them. Otherwise profit is derived only when usable revenue and expense values both exist.
- KPIs and reports label unavailable measures rather than presenting unsupported zeroes as calculated facts.
- Margin is total profit divided by total revenue when both are available and revenue is non-zero.
- MoM growth compares the latest two months present in the data. It is unavailable when the prior month has zero revenue.
- Currency is not inferable from the provided column schema. Select the display currency in the sidebar to match your source data; the default is INR.

## AI data handling and voice requirements

Gemini is optional. When enabled, the app sends the user's question and aggregated dashboard metrics to Google's Gemini service; it does not send the uploaded row-level dataset. The same aggregate-only behavior is used for generated AI insights. Do not use the Gemini option with data you are not authorized to share externally. Without an API key, local analytics remains available.

Voice input uses the browser's Web Speech API and requires a supported browser plus microphone permission. Final recognized speech is sent to the dashboard and placed in the editable question field; review it and click **Ask AI** to submit it. Some browsers rely on an internet speech-recognition service. If unsupported or blocked, the text input continues to work. Answer playback uses the browser's SpeechSynthesis API and only starts when the user clicks **Read answer**.

The current project does not include a database connection or SQL execution layer. Uploaded CSV/Excel data is analyzed locally with Pandas; no SQL is generated or executed.

## Setup on Windows and VS Code

1. Install Python 3.11 or newer and VS Code. In VS Code, install the Microsoft Python extension.
2. Open the `automated-insight-generator` folder in VS Code (`File` → `Open Folder...`).
3. Open the integrated terminal (`Terminal` → `New Terminal`) and create a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   ```

4. Activate it in PowerShell:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, either use the VS Code Command Prompt terminal and run `.\.venv\Scripts\activate.bat`, or allow scripts for the current user with `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`.

5. Install dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

6. To enable AI insights, copy `.env.example` to `.env` and set your key:

   ```powershell
   Copy-Item .env.example .env
   ```

   Edit `.env` and replace the example value with your Google Gemini API key. Keep `.env` private; it is excluded by `.gitignore`. The dashboard, local Q&A, local insights, and reports work without a key.

7. Run the dashboard from the project folder:

   ```powershell
   streamlit run app.py
   ```

   Streamlit opens the app in your browser. Upload your own CSV/Excel file or choose **Load the included sample dataset**.

8. Run the tests:

   ```powershell
   python -m pytest -q
   ```

## Data input notes

CSV files should be UTF-8 when possible. Excel files are read from the first worksheet. Invalid dates are excluded from date-based trend calculations; invalid numeric cells are treated as missing. A file with no recognized business metric or no valid metric values is rejected with a visible validation message.

## Project layout

```text
automated-insight-generator/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
├── src/
│   ├── dataloader.py
│   ├── dataanalyzer.py
│   ├── insightgenerator.py
│   ├── reportgenerator.py
│   └── voice_component/
│       └── index.html
├── tests/
│   ├── testdataloader.py
│   ├── testdataanalyzer.py
│   ├── testinsightgenerator.py
│   └── testreportgenerator.py
└── sampledata/
    └── salesdata.csv
```
