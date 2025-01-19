import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
from pypdf import PdfReader
import pandas as pd
from typing import Dict, List, Tuple
import json
from dataclasses import dataclass
import os
from dotenv import load_dotenv

import sys

@dataclass
class RiskItem:
    category: str
    description: str
    severity: str
    contract_section: str
    relevant_text: str
    mitigation_suggestion: str

class SimpleVectorStore:
    def __init__(self):
        # Initialize the sentence transformer model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.texts = []
        self.index = None
        
    def add_texts(self, texts: List[str]):
        """Add texts to the vector store"""
        self.texts = texts
        
        # Create embeddings
        embeddings = self.model.encode(texts)
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        
        # Add vectors to the index
        self.index.add(np.array(embeddings).astype('float32'))
    
    def search(self, query: str, k: int = 3) -> List[str]:
        """Search for similar texts"""
        # Create query embedding
        query_vector = self.model.encode([query])
        
        # Search in FAISS
        distances, indices = self.index.search(
            np.array(query_vector).astype('float32'), 
            k
        )
        
        # Return found texts
        return [self.texts[idx] for idx in indices[0]]

class PPARiskAnalyzer:
    def __init__(self, gemini_api_key: str):
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.vector_store = SimpleVectorStore()
        
        self.risk_categories = [
            "financial",
            "governance",
            "sentiment"
        ]
        
        self.risk_queries = {
            "financial": [
                "price adjustment mechanisms and risks",
                "payment terms and credit risks",
                "financial security and collateral requirements"
            ],
            "governance": [
                "change in law provisions and regulatory risks",
                "reporting and compliance requirements",
                "decision-making and approval processes"
            ],
            "sentiment": [
                "force majeure definitions and implications",
                "default triggers and conditions",
                "dispute resolution mechanisms"
            ]
        }

    def load_document(self, pdf_path: str) -> int:
        """Load and process PDF document"""
        chunks = []
        
        try:
            # Read PDF
            reader = PdfReader(pdf_path)
            
            # Process each page
            for page in reader.pages:
                text = page.extract_text()
                # Split into chunks
                page_chunks = self._split_text(text)
                chunks.extend(page_chunks)
            
            # Store in vector store
            self.vector_store.add_texts(chunks)
            
            return len(chunks)
            
        except Exception as e:
            print(f"Error loading document: {str(e)}")
            raise

    def _split_text(self, text: str, chunk_size: int = 5000) -> List[str]:
        """Split text into chunks"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            if current_size + len(word) > chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_size = len(word)
            else:
                current_chunk.append(word)
                current_size += len(word)
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def analyze_risks(self) -> List[RiskItem]:
        """Analyze document for risks"""
        risks = []
        
        try:
            for category in self.risk_categories:
                for query in self.risk_queries[category]:
                    # Get relevant chunks
                    relevant_chunks = self.vector_store.search(
                        query=query,
                        k=2
                    )
                    
                    # Combine chunks for analysis
                    combined_text = "\n".join(relevant_chunks)
                    
                    # Generate analysis
                    response = self.model.generate_content(
                        self._create_risk_prompt(category, query, combined_text)
                    )
                    
                    # Parse response
                    try:
                        # Clean and extract JSON content
                        response_text = response.text.strip()
                        
                        # Find the JSON content between markdown markers if present
                        if '```json' in response_text and '```' in response_text:
                            start = response_text.find('```json') + 7
                            end = response_text.rfind('```')
                            response_text = response_text[start:end]
                        
                        # Clean up any remaining whitespace
                        response_text = response_text.strip()
                        
                        # Parse the cleaned JSON
                        risk_data = json.loads(response_text)
                        
                        for risk in risk_data.get('risks', []):
                            risks.append(
                                RiskItem(
                                    category=category,
                                    description=risk['description'],
                                    severity=risk['severity'],
                                    contract_section=risk['contract_section'],
                                    relevant_text=risk['relevant_text'],
                                    mitigation_suggestion=risk['mitigation_suggestion']
                                )
                            )
                    except json.JSONDecodeError as e:
                        print(f"Error parsing response for {category} - {query}. Error: {str(e)}")
                        print(f"Response text: {response_text}")
                        continue
                        
        except Exception as e:
            print(f"Error in risk analysis: {str(e)}")
            raise
            
        return risks

    def _create_risk_prompt(self, category: str, query: str, text: str) -> str:
        """Create prompt for risk analysis"""
        return f"""
        Analyze the following Power Purchase Agreement (PPA) contract text for {category} risks,
        specifically focusing on {query}.
        
        Provide your analysis in JSON format with the following structure:
        {{
            "risks": [
                {{
                    "description": "Description of the risk",
                    "severity": "HIGH/MEDIUM/LOW",
                    "contract_section": "Relevant section name",
                    "relevant_text": "Specific text that indicates the risk",
                    "mitigation_suggestion": "Suggested mitigation strategy"
                }}
            ]
        }}
        
        Contract text:
        {text}
        
        Provide only the JSON response without any additional text.
        """

    def generate_risk_report(self, risks: List[RiskItem]) -> Dict:
        """Generate risk report"""
        report = {
            'summary': {
                'total_risks': len(risks),
                'high_severity': len([r for r in risks if r.severity == 'HIGH']),
                'medium_severity': len([r for r in risks if r.severity == 'MEDIUM']),
                'low_severity': len([r for r in risks if r.severity == 'LOW'])
            },
            'risks_by_category': {}
        }
        
        # Organize by category
        for category in self.risk_categories:
            category_risks = [r for r in risks if r.category == category]
            report['risks_by_category'][category] = {
                'count': len(category_risks),
                'risks': [
                    {
                        'description': r.description,
                        'severity': r.severity,
                        'contract_section': r.contract_section,
                        'relevant_text': r.relevant_text,
                        'mitigation': r.mitigation_suggestion
                    }
                    for r in category_risks
                ]
            }
        
        return report

def analyze_document_risks(pdf_path: str, gemini_api_key: str="") -> dict:

    load_dotenv()
    gemini_api_key=os.getenv("GOOGLE_API_KEY")

    try:
        # Initialize analyzer
        analyzer = PPARiskAnalyzer(gemini_api_key=gemini_api_key)
        
        # Load document
        num_chunks = analyzer.load_document(pdf_path)
        print(f"Loaded {num_chunks} document chunks")
        
        # Analyze risks
        risks = analyzer.analyze_risks()
        
        # Generate report
        report = analyzer.generate_risk_report(risks)
        print(report)
        return report
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python -m climate-hackathon.pdf_parser <pdf_path>")
    else:
        pdf_path = sys.argv[1]
        result = analyze_document_risks(pdf_path)
        print(result)
        
# def main():
#     try:
#         # Load environment variables
#         load_dotenv()
        
#         # Initialize analyzer
#         analyzer = PPARiskAnalyzer(gemini_api_key=os.getenv("GOOGLE_API_KEY"))
        
#         # Load document
#         num_chunks = analyzer.load_document("PPA.pdf")
#         print(f"Loaded {num_chunks} document chunks")
        
#         # Analyze risks - remove await
#         risks = analyzer.analyze_risks()
        
#         # Generate report
#         report = analyzer.generate_risk_report(risks)
        
#         # Save report
#         with open('risk_report.json', 'w') as f:
#             json.dump(report, f, indent=2)
            
#     except Exception as e:
#         print(f"Error in main execution: {str(e)}")
#         raise

# if __name__ == "__main__":
#     main()  # Remove asyncio.run()