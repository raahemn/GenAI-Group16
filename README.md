# GenAI-Group16: RAG Model for Speech and Language Processing
Welcome to the repository for GenAI-Group16, where we have developed a Retrieval-Augmented Generation (RAG) model for the course on Speech and Language Processing for Generative AI. This project aims to enhance course recommendation systems by integrating state-of-the-art AI techniques.

### Project Overview
Our RAG model leverages the power of large language models and vector embeddings to provide precise course recommendations based on a student’s academic profile and interests. This system not only suggests courses but also predicts workload and provides tailored self-study resources.

### Features
Personalized Course Recommendations: Utilizes student data to suggest suitable courses.
Workload Prediction: Estimates the potential workload for recommended courses.
Resource Retrieval: Automatically identifies and retrieves relevant self-study materials.

# Technical Description
### Model Architecture
Our project employs a Retrieval-Augmented Generation (RAG) architecture, which combines the benefits of retrieval-based and generative AI models to enhance the relevance and accuracy of course recommendations.

### Vector Embedding
The model uses Ollama Embeddings from the LangChain community for vector embedding. These embeddings are crucial for transforming textual data into numerical data that can be efficiently processed and compared by our model.

### Vector Store
We utilize Chroma, a vector storage solution from the LangChain community, to manage and retrieve vector embeddings effectively. Chroma supports efficient querying and scalability, essential for handling our dataset.

### Language Model (LLM)
The backbone of our generative capabilities is the HuggingFace Hub LLM. This pre-trained model facilitates the generation of text and enables the nuanced understanding of complex queries and materials.

### Dataset
Our dataset comprises diverse course outlines and student feedback collected from multiple academic institutions. It includes detailed annotations on course content, difficulty levels, and student outcomes to train our model comprehensively.

# Getting Started
### Prerequisites
 - Python 3.8+
 - pip
 - virtualenv (optional)

### Installation
 - Clone the repository and install the required packages:

 - git clone https://github.com/YourRepository/GenAI-Group16.git
 - cd GenAI-Group16
 - pip install -r requirements.txt


### Running the Model
To start the model and access the course recommendation system, execute:
- python run_model.py





