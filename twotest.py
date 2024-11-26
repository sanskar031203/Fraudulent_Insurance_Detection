from groq import Groq
import json
import fitz  # PyMuPDF
import os
from extra_copy import datat

class MedicalReportAnalyzer:
    def __init__(self, groq_api_key):
        """
        Initialize the Medical Report Analyzer with Groq API key
        
        :param groq_api_key: Your Groq API key for AI processing
        """
        self.client = Groq(api_key=groq_api_key)
        self.symptoms = [
            'age',
            'gender',
            'dizziness',
            'disorientation',
            'low_bp',
            'severity_score',
            'lab_results',
            'prior_admissions',
            'outcome_severity',
            'admission_valid'
        ]
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text content from PDF file using PyMuPDF.
        
        :param pdf_path: Path to the PDF file
        :return: Extracted text content
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
            
        try:
            # Open the PDF
            doc = fitz.open(pdf_path)
            
            # Extract text from all pages
            text = ""
            for page in doc:
                text += page.get_text()
            
            # Close the document
            doc.close()
            
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")

    def analyze_medical_report(self, medical_text):
        """
        Analyze medical report text and extract detailed medical information
        
        :param medical_text: Text of the medical report
        :return: Dictionary of medical information
        """
        prompt = f"""
        Carefully analyze the following medical report text and extract the following comprehensive information:
        - Patient Age
        - Patient Gender
        - Dizziness (binary: 0 or 1)
        - Disorientation (binary: 0 or 1)
        - Low Blood Pressure (binary: 0 or 1)
        - Severity Score (sum of Dizziness, Disorientation, Low BP)
        - Lab Results (numeric scale from 0 to 10)
        - Prior Admissions (numeric count from 0 to 5)
        - Outcome Severity (numeric scale from 0 to 10)
        - Admission Validity (1 if Severity Score > 0 and Outcome Severity > 5, else 0)

        Respond in a JSON format with detailed information.
        Example output: {{
            "age": 45,
            "gender": "female",
            "dizziness": 1,
            "disorientation": 0,
            "low_bp": 1,
            "severity_score": 2,
            "lab_results": 7,
            "prior_admissions": 2,
            "outcome_severity": 6,
            "admission_valid": 1
        }}

        Guidelines:
        - Use binary values (0 or 1) for symptoms
        - Calculate severity score by summing dizziness, disorientation, and low BP
        - Determine admission validity based on severity score and outcome severity
        - If information is not explicitly stated, make a reasonable inference
        - Ensure all numeric values fall within the specified ranges

        Medical Report Text:
        {medical_text}
        """
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama3-8b-8192",
                response_format={"type": "json_object"},
                max_tokens=300
            )
            
            # Parse the response
            response = json.loads(chat_completion.choices[0].message.content)
            return response
        
        except Exception as e:
            return f"Error processing report: {str(e)}"

    def paragraphing_medical_report(self, medical_text):
        """
        Analyze medical report text and extract detailed medical information
        
        :param medical_text: Text of the medical report
        :return: Dictionary of medical information
        """
        prompt = f"""
        Please analyze this medical admission record {medical_text} which is in json and provide explanation
        Pay special attention to the relationships between symptoms, severity scores, and outcomes.

        Focus on following points : 
        Patient presents with dizziness but no disorientation
        Has low blood pressure
        Overall severity score of 2 out of 10
        Lab results value of 7
        2 prior hospital admissions
        Outcome severity rated as 6 out of 10
        
        Based on these data points, please:
        
        Explain any inconsistencies or red flags in the relationship between symptoms and severity scores
        Analyze whether the outcome severity (6) aligns with the initial severity score (2)
        Evaluate if the lab results value is reasonable given the symptoms
        Consider if the prior admission history supports or contradicts the current presentation
        
        Please structure your response as a coherent paragraph that a medical auditor could use in their review
        
        please return a single paragraph in string format
        """
        
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model="llama3-8b-8192",
                max_tokens=300
            )
            
            # Parse the response
            response = chat_completion.choices[0].message.content
            return response
        
        except Exception as e:
            return f"Error processing report: {str(e)}" 


def process_medical_reports(api_key, pdf_path):
    """
    Process medical reports from a PDF file and print their symptom analysis
    
    :param api_key: Groq API key
    :param pdf_path: Path to the PDF file containing medical reports
    :return: Analysis results
    """
    # Create analyzer instance
    analyzer = MedicalReportAnalyzer(api_key)
    
    try:
        # Extract text from PDF
        medical_text = analyzer.extract_text_from_pdf(pdf_path)
        print("\nExtracted text from PDF:")
        print(medical_text)
        
        # Analyze the medical report
        print("\nMedical Report Analysis:")
        result = analyzer.analyze_medical_report(medical_text)
        print(json.dumps(result, indent=2))


        return result
        
    except Exception as e:
        print(f"Error processing report: {e}")
        return None

def prepare_sample_reports():
    """
    Prepare a list of sample medical reports for analysis
    
    :return: List of medical report texts
    """
    sample_reports = [
        """
        45-year-old female patient presented with acute symptoms.
        Experiencing significant lightheadedness and disorientation. 
        Blood pressure critically low at 90/60 mmHg.
        Complex medical history with multiple prior hospitalizations.
        Lab tests indicate potentially serious underlying condition.
        Patient's current state suggests high-risk medical scenario.
        Recommended immediate medical intervention.
        """,
        
        """
        32-year-old male patient for routine check-up.
        Mild symptoms of uneasiness observed.
        Stable vital signs and clear mental state.
        No significant medical concerns detected.
        Minimal lab test variations within normal range.
        No prior hospital admissions.
        """,
        
        """
        58-year-old male with severe medical complications.
        Pronounced dizziness and significant disorientation.
        Blood pressure fluctuations and unstable condition.
        Multiple previous hospitalizations for chronic conditions.
        Extensive lab work shows critical health markers.
        Immediate intensive care intervention required.
        """
    ]
    return sample_reports
def admi(PDF_PATH):
    """
    Main function to orchestrate the medical report analysis
    """
    GROQ_API_KEY = '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'
    analyzer = MedicalReportAnalyzer(GROQ_API_KEY)

    try:
        result = process_medical_reports(GROQ_API_KEY, PDF_PATH)
        if result:
            s = datat(result)
            result = analyzer.paragraphing_medical_report(s)
            return result,s['fraud_probability']
    except Exception as e:
        print(f"An error occurred: {e}")