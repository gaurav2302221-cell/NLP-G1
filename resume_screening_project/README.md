# Automated Resume Screening System using NLP

## 📋 Project Overview

An advanced, intelligent resume screening system that automatically analyzes and ranks candidates based on how well their resumes match a given job description. The system uses state-of-the-art NLP techniques including Named Entity Recognition (NER), TF-IDF, BERT embeddings, and skill matching to provide comprehensive candidate evaluation.

### Key Features:
- **Multi-format PDF resume parsing** with robust error handling
- **Advanced text preprocessing** using spaCy and NLTK
- **Skill extraction** using predefined skill dictionary (20+ categories)
- **Three-tier similarity calculation** (TF-IDF, BERT, Word Embeddings)
- **Candidate ranking** based on weighted scoring (70% similarity + 30% skill match)
- **Interactive Streamlit dashboard** for easy candidate management
- **Model explainability** using SHAP for transparency
- **Evaluation metrics** (Precision, Recall, F1 Score)

---

## 🏗️ Architecture

### System Components:

```
resume_screening_project/
├── data/
│   └── resumes/              # Directory for storing resume PDFs
├── models/                   # Pre-trained model storage
├── src/
│   ├── __init__.py          # Package initialization
│   ├── resume_parser.py     # PDF resume extraction
│   ├── text_preprocessing.py # NLP text cleaning & normalization
│   ├── skill_extractor.py   # Skill detection & matching
│   ├── job_parser.py        # Job description processing
│   ├── similarity_model.py  # Multi-method similarity scoring
│   ├── ranking_engine.py    # Candidate ranking & scoring
│   └── evaluation.py        # Model evaluation metrics
├── app.py                   # Streamlit web application
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

### Module Descriptions:

| Module | Purpose |
|--------|---------|
| `resume_parser.py` | Extracts text from PDF files using PyMuPDF |
| `text_preprocessing.py` | Tokenization, lemmatization, stopword removal |
| `skill_extractor.py` | Pattern matching for 20+ technical skills |
| `job_parser.py` | Parses job requirements and extracts skills |
| `similarity_model.py` | TF-IDF, BERT, and embedding-based similarity |
| `ranking_engine.py` | Combines scores for final candidate ranking |
| `evaluation.py` | Precision, Recall, F1 Score calculation |

---

## 🔧 Installation

### Prerequisites:
- Python 3.8 or higher
- pip or conda for package management
- 4GB+ RAM recommended for BERT models

### Step 1: Clone or Create Project Directory

```bash
cd c:\Users\hp\Desktop\NLPpro
cd resume_screening_project
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### Step 5: Verify Installation

```bash
python -c "import spacy; import streamlit; print('All dependencies installed successfully!')"
```

---

## 🚀 Running the Application

### Start the Streamlit Dashboard:

```bash
# From the project root directory
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Using the Application:

1. **Upload Resumes**: Click "Browse files" to upload one or multiple PDF resumes
2. **Enter Job Description**: Paste the job description in the text area
3. **Click "Analyze Resumes"**: The system will process and rank candidates
4. **View Results**: 
   - See candidate rankings in the table
   - View skill matching details
   - Analyze similarity score breakdowns
   - Export results if needed

---

## 📊 How It Works

### 1. Resume Processing
- PDFs are parsed using PyMuPDF
- Text is extracted with error handling
- Candidate name is extracted from filename

### 2. Text Preprocessing
```
Raw Text → Lowercase → Remove URLs/Emails → Remove Special Chars → 
Tokenize → Remove Stopwords → Lemmatize
```

### 3. Information Extraction
**Named Entity Recognition**: Company names, organizations, locations  
**Skill Extraction**: Matches 50+ technical skills from predefined dictionary

### 4. Similarity Calculation
Three parallel approaches averaged together:

- **TF-IDF + Cosine Similarity** (0-100%)
  - Measures term frequency importance
  - Fast computation
  
- **BERT Embeddings** (0-100%)
  - Uses `all-MiniLM-L6-v2` model
  - Captures semantic meaning
  - State-of-the-art performance
  
- **Word Embeddings** (0-100%)
  - Average word vector similarity
  - Captures contextual relationships

**Final Score = (TF-IDF + BERT + Word Embeddings) / 3**

### 5. Candidate Ranking

```
Final Rank Score = (Similarity Score × 0.70) + (Skill Match % × 0.30)
```

This dual-weighted approach ensures both textual relevance and technical skill alignment.

### 6. Skill Matching

```
Skill Match % = (Matched Skills / Required Skills) × 100
```

Shows which required skills candidates possess and which are missing.

---

## 📈 Supported Skills

### Programming Languages
Python, Java, C++, JavaScript, C#, PHP, Ruby, Go, Rust, Kotlin, Swift

### Machine Learning
Machine Learning, Deep Learning, NLP, Computer Vision, Neural Networks

### Data Science
Data Science, Data Analysis, Statistics, Big Data, Hadoop, Spark

### Databases
SQL, MongoDB, Cassandra, Redis, DynamoDB, Elasticsearch

### Frameworks & Tools
Django, Flask, FastAPI, Spring, React, Angular, Vue

### Cloud Platforms
AWS, Azure, GCP, Heroku, IBM Cloud

### Containerization
Docker, Kubernetes, Docker Compose

### Other
Git, Jupyter, Tableau, PowerBI, Jenkins, GitHub Actions, and more!

---

## 📤 Output & Visualization

### Ranking Table
Displays all candidates with:
- Rank number
- Candidate name
- Similarity score (0-100)
- Skill match percentage
- Final combined score

### Skill Analysis
- Matching skills (possessed by candidate)
- Missing skills (required but not present)
- Additional skills (candidate has but not required)

### Visualization Charts
- Bar chart of similarity scores
- Bar chart of skill match percentages
- Overall score distribution

### Top Candidate Highlight
The highest-ranked candidate is highlighted with special emphasis on:
- Matching strengths
- Areas for improvement
- Overall fit assessment

---

## 🎯 Evaluation Metrics

The system provides standard ML evaluation metrics:

- **Precision**: Of predicted matches, how many were correct?
- **Recall**: Of actual matches, how many were found?
- **F1 Score**: Harmonic mean of precision and recall
- **Accuracy**: Overall correctness rate

These help validate the system's performance over time.

---

## 📝 Usage Examples

### Example 1: Data Science Position

**Job Description Excerpt:**
```
Seeking talented Data Scientist
- 5+ years experience with Python, SQL
- Strong background in Machine Learning
- Experience with TensorFlow and PyTorch
- AWS and Docker knowledge preferred
```

**Resume Analysis:**
- Extracts skills: Python, Machine Learning, TensorFlow, AWS
- Calculates skill match: 75% (3 out of 4 required skills)
- Computes similarity: 82% (based on BERT, TF-IDF, embeddings)
- Final rank score: 79.25% (70% × 0.82 + 30% × 0.75)

### Example 2: Multiple Candidates

**Input:** 5 resume PDFs + Job Description  
**Output:** Ranked list with:
```
Rank 1: John_Doe.pdf - Score: 85.4%
Rank 2: Jane_Smith.pdf - Score: 78.2%
Rank 3: Bob_Johnson.pdf - Score: 72.1%
Rank 4: Alice_Brown.pdf - Score: 68.9%
Rank 5: Charlie_White.pdf - Score: 61.2%
```

---

## 🔍 Explainability

### SHAP Integration
SHAP (SHapley Additive exPlanations) values explain:
- Which resume features drove similarity scores
- Feature importance in matching decision
- Individual feature contributions

### Interpretable Output
Clear explanation of why resumes ranked as they did:
- Skill matching breakdown
- Similar key phrases found
- Missing critical requirements

---

## ⚙️ Configuration

### Adjusting Weights

In `ranking_engine.py`, modify:
```python
rank_candidates(
    candidate_resumes,
    job_description,
    job_skills,
    similarity_weight=0.7,  # Change this
    skill_weight=0.3        # Or this
)
```

### Adding Skills

Edit `SKILL_DICTIONARY` in `skill_extractor.py`:
```python
SKILL_DICTIONARY = {
    'new_skill': ['variation1', 'variation2'],
    # Add more...
}
```

### Filtering by Score

In app.py, filter qualified candidates:
```python
qualified = filter_candidates_by_threshold(ranked_candidates, threshold=70.0)
```

---

## 🧪 Testing

### Test with Sample Resumes

1. Create sample PDF resumes in `data/resumes/`
2. Create a test job description
3. Run the application and analyze

### Evaluate Model Performance

```python
from src.evaluation import evaluate_model

true_labels = [1, 0, 1, 0, 1]  # 1 = good match, 0 = not match
predictions = [1, 0, 1, 1, 1]

metrics = evaluate_model(true_labels, predictions)
print(f"Precision: {metrics['precision']}")
print(f"Recall: {metrics['recall']}")
print(f"F1 Score: {metrics['f1_score']}")
```

---

## 🐛 Troubleshooting

### Issue: spaCy model not found
```bash
python -m spacy download en_core_web_sm
```

### Issue: BERT model download issues
- First run may take 2-3 minutes to download model
- Check internet connection
- Clear cache: Delete `.cache/huggingface/`

### Issue: Memory error with large documents
- Streamlit resets with each upload
- For batch processing, optimize in code directly
- Use smaller sample resumes for testing

### Issue: PDF parsing fails
- Ensure PDFs are text-based (not scanned images)
- Try with different PDF file
- Check file is not corrupted

---

## 📊 Performance Metrics

| Component | Time (Sec) | Notes |
|-----------|-----------|-------|
| Resume Parsing | 0.1-0.5 | Per PDF |
| Text Preprocessing | 0.05-0.2 | Per resume |
| Skill Extraction | 0.02-0.1 | Pattern matching |
| BERT Similarity | 0.5-2.0 | First load slower |
| Total (Single Resume) | ~3-4 | Includes model init |

---

## 📚 Tech Stack Details

| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.8+ | Core language |
| spaCy | 3.7.2 | NLP & NER |
| NLTK | 3.8.1 | Tokenization & lemmatization |
| scikit-learn | 1.3.2 | TF-IDF & cosine similarity |
| sentence-transformers | 2.2.2 | BERT embeddings |
| Streamlit | 1.28.1 | Web UI |
| PyMuPDF | 1.23.8 | PDF parsing |
| SHAP | 0.42.1 | Model explainability |
| LIME | 0.2.0 | Local interpretability |

---

## 🎓 Learning Resources

- [spaCy Documentation](https://spacy.io/)
- [Sentence Transformers](https://www.sbert.net/)
- [SHAP GitHub](https://github.com/slundberg/shap)
- [Streamlit Docs](https://docs.streamlit.io/)
- [Scikit-learn ML Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

## 📄 License

This project is open-source and available for educational and commercial use.

---

## 👥 Author Information

**Project:** Automated Resume Screening System using NLP  
**Version:** 1.0.0  
**Updated:** February 2026  
**Created by:** AI/ML Engineering Team

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Create a feature branch
2. Implement changes with tests
3. Submit pull request with description
4. Ensure code quality and documentation

---

## 📧 Support

For issues, questions, or improvements:
- Check the Troubleshooting section
- Review code comments and docstrings
- Verify all dependencies are installed
- Test with simpler inputs first

---

## ✨ Future Enhancements

- [ ] Resume database integration
- [ ] Machine learning model fine-tuning
- [ ] Real-time skill updates
- [ ] Multi-language support
- [ ] Advanced visualization dashboards
- [ ] API endpoint development
- [ ] Batch processing pipeline
- [ ] Resume template parsing
- [ ] Candidate communication integration
- [ ] Performance analytics & reporting

---

## ⚖️ Disclaimer

This system is designed to assist in resume screening and should not be the sole basis for hiring decisions. Human review and consideration of factors beyond resume content are essential for fair and comprehensive candidate evaluation.

---

**Happy screening! 🎯**
