import streamlit as st
import os
import json
import pandas as pd
import tempfile
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Data Science Job Application Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create directories for uploads
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("data/uploads"):
    os.makedirs("data/uploads")

# Resume Parser Class
class ResumeParser:
    def parse_resume(self, file_path):
        """Parse resume and extract information"""
        # In a real implementation, this would use libraries like PyPDF2, python-docx, and spaCy
        # For this demo, we'll simulate parsing with sample data
        
        file_extension = file_path.split('.')[-1].lower()
        
        # Simulate different parsing based on file type
        if file_extension in ['pdf', 'doc', 'docx']:
            # Generate sample parsed data
            parsed_data = {
                "contact_info": {
                    "name": "Sample Candidate",
                    "email": "candidate@example.com",
                    "phone": "+91 9876543210",
                    "location": "Bangalore, Karnataka"
                },
                "education": "B.Tech in Computer Science from IIT Delhi",
                "experience": "Data Scientist at ABC Analytics for 3 years",
                "experience_years": 3,
                "skills": ["Python", "Machine Learning", "Data Analysis", "SQL", "TensorFlow"],
                "keyword_matches": {
                    "python": 5,
                    "machine learning": 4,
                    "data analysis": 4,
                    "sql": 3,
                    "tensorflow": 2,
                    "data visualization": 2,
                    "statistics": 3,
                    "deep learning": 1
                },
                "resume_text": f"Sample resume text for a data scientist with experience in Python, Machine Learning, and Data Analysis. The candidate has worked at ABC Analytics for 3 years and has a B.Tech in Computer Science from IIT Delhi."
            }
            return parsed_data
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

# ATS Scorer Class
class ATSScorer:
    def score_resume_for_job(self, resume_text, job_description):
        """Score resume against job description"""
        # In a real implementation, this would use NLP techniques
        # For this demo, we'll simulate scoring
        
        # Extract keywords from job description (simplified)
        job_keywords = ["python", "machine learning", "data analysis", "sql", "tensorflow", 
                       "data visualization", "statistics", "deep learning", "communication"]
        
        # Count keyword occurrences in resume (simplified)
        keyword_counts = {}
        for keyword in job_keywords:
            if keyword in resume_text.lower():
                # Simulate different counts
                import random
                keyword_counts[keyword] = random.randint(1, 5)
        
        # Calculate match percentage
        total_keywords = len(job_keywords)
        matched_keywords = len(keyword_counts)
        match_percentage = (matched_keywords / total_keywords) * 100
        
        # Generate improvement suggestions
        missing_keywords = [k for k in job_keywords if k not in keyword_counts]
        suggestions = []
        if missing_keywords:
            suggestions.append(f"Add these missing keywords: {', '.join(missing_keywords)}")
        
        if match_percentage < 70:
            suggestions.append("Tailor your resume more specifically to the job description")
        
        # Generate section scores
        section_scores = {
            "skills": min(100, match_percentage + 10),
            "experience": min(100, match_percentage - 5),
            "education": min(100, match_percentage + 5),
            "overall": match_percentage
        }
        
        return {
            "match_percentage": match_percentage,
            "keyword_matches": keyword_counts,
            "missing_keywords": missing_keywords,
            "section_scores": section_scores,
            "improvement_suggestions": suggestions
        }

# Cover Letter Generator Class
class CoverLetterGenerator:
    def __init__(self):
        # Templates for different parts of the cover letter
        # Fresher intro templates
        self.fresher_intro_templates = [
            "As a recent graduate with a strong foundation in {background}, I am excited to apply for the {job_title} position at {company_name}.",
            "I am writing to express my interest in the {job_title} role at {company_name}. With my educational background in {background} and passion for data science, I am eager to begin my professional journey with your team.",
            "I recently completed my degree in {education} and am enthusiastic about applying for the {job_title} position at {company_name}. My academic focus on {background} has prepared me well for this role."
        ]
        
        # Experienced intro templates
        self.experienced_intro_templates = [
            "With {experience_years}+ years of experience in {background}, I am eager to bring my expertise to the {job_title} position at {company_name}.",
            "I am writing to express my interest in the {job_title} position at {company_name}. My {experience_years}+ years of experience in {background} align perfectly with the requirements outlined in your job posting.",
            "As an experienced professional with expertise in {background} spanning {experience_years}+ years, I am thrilled to submit my application for the {job_title} position at {company_name}."
        ]
        
        # Fresher body templates
        self.fresher_body_templates = [
            "During my academic career, I developed strong skills in {skills_list}. In my {project_type}, I {achievement_1}. Additionally, I {achievement_2}, which demonstrates my ability to {relevant_skill}.",
            "My educational background has equipped me with knowledge in {skills_list}. For my {project_type}, I successfully {achievement_1}. I also {achievement_2}, showcasing my aptitude for {relevant_skill}."
        ]
        
        # Experienced body templates
        self.experienced_body_templates = [
            "Throughout my career, I have developed strong skills in {skills_list}. At {previous_company}, I {achievement_1}. Additionally, I {achievement_2}, which demonstrates my ability to {relevant_skill}.",
            "My professional experience includes working with {skills_list} to solve complex data problems. While at {previous_company}, I successfully {achievement_1}. I also {achievement_2}, showcasing my expertise in {relevant_skill}."
        ]
        
        # Common company paragraph templates
        self.company_paragraph_templates = [
            "I am particularly drawn to {company_name} because of your {company_value}. Your work on {company_project} resonates with my professional interests, and I am excited about the opportunity to contribute to {future_goal}.",
            "What attracts me to {company_name} is your commitment to {company_value}. I've been following your progress on {company_project}, and I'm enthusiastic about the prospect of helping {future_goal}."
        ]
        
        # Fresher closing templates
        self.fresher_closing_templates = [
            "Thank you for considering my application. I am excited about the possibility of beginning my career at {company_name} and would welcome the opportunity to discuss how my academic background and skills would be a good match for this position. I look forward to hearing from you.",
            "I would appreciate the chance to further discuss how my educational background and skills could benefit {company_name}. Thank you for your time and consideration, and I look forward to the possibility of starting my professional journey with your team."
        ]
        
        # Experienced closing templates
        self.experienced_closing_templates = [
            "Thank you for considering my application. I am excited about the possibility of joining {company_name} and would welcome the opportunity to discuss how my background and skills would be a good match for this position. I look forward to hearing from you.",
            "I would appreciate the chance to further discuss how my skills and experience would benefit {company_name}. Thank you for your time and consideration, and I look forward to the possibility of working together."
        ]
        
        # Project types for freshers
        self.project_types = [
            "capstone project",
            "thesis research",
            "course project",
            "internship",
            "hackathon project"
        ]
        
        # Company values for data science/tech companies
        self.company_values = [
            "innovation and cutting-edge technology",
            "data-driven decision making",
            "solving complex business problems through analytics",
            "creating AI solutions that positively impact society",
            "fostering a collaborative and inclusive work environment"
        ]
        
        # Company projects for tech/data science companies
        self.company_projects = [
            "developing scalable machine learning infrastructure",
            "implementing AI-driven customer insights",
            "creating advanced analytics platforms",
            "building real-time recommendation systems",
            "developing natural language processing solutions"
        ]
        
        # Future goals for tech/data science companies
        self.future_goals = [
            "drive innovation through data-driven insights",
            "scale your AI capabilities across the organization",
            "develop next-generation machine learning solutions",
            "improve customer experiences through predictive analytics",
            "optimize operational efficiency using data science"
        ]
    
    def generate_cover_letter(self, resume_data, job_data):
        """
        Generate a customized cover letter based on resume data and job data
        """
        import random
        
        # Extract necessary information
        job_title = job_data.get('title', 'Data Science position')
        company_name = job_data.get('company', 'your company')
        job_description = job_data.get('description', '')
        
        # Extract or generate candidate information from resume
        candidate_name = resume_data.get('contact_info', {}).get('name', 'Candidate Name')
        candidate_email = resume_data.get('contact_info', {}).get('email', 'candidate@example.com')
        candidate_phone = resume_data.get('contact_info', {}).get('phone', '+91 XXXXXXXXXX')
        candidate_location = resume_data.get('contact_info', {}).get('location', 'City, India')
        
        # Determine background based on resume skills
        skills = list(resume_data.get('keyword_matches', {}).keys())
        background = "data science and machine learning"
        
        # Determine experience years (estimate or from resume)
        experience_years = resume_data.get('experience_years', 0)
        
        # Determine if candidate is a fresher or experienced
        is_fresher = experience_years < 2
        
        # Get education information
        education = resume_data.get('education', 'Computer Science or related field')
        
        # Generate skills list for body paragraph
        skills_list = ", ".join(skills[:3]) if skills else "data analysis, machine learning, and Python"
        
        # Generate achievements
        achievements = [
            "developed a machine learning model that achieved 85% accuracy in predicting outcomes",
            "created a data visualization dashboard that effectively communicated complex findings",
            "implemented a classification algorithm that correctly categorized data with 90% precision",
            "designed and executed a data analysis project that revealed valuable insights"
        ]
        achievement_1 = random.choice(achievements)
        achievement_2 = random.choice([a for a in achievements if a != achievement_1])
        
        # Determine previous company or project type
        if is_fresher:
            project_type = random.choice(self.project_types)
            previous_company = None
        else:
            previous_company = "my previous organization"
            if "experience" in resume_data:
                exp_text = resume_data["experience"]
                if "at" in exp_text:
                    previous_company = exp_text.split("at")[1].strip().split(" ")[0]
            project_type = None
        
        # Determine relevant skill
        relevant_skill = "solve complex data problems efficiently"
        
        # Generate company paragraph details
        company_value = random.choice(self.company_values)
        company_project = random.choice(self.company_projects)
        future_goal = random.choice(self.future_goals)
        
        # Format the date
        current_date = datetime.now().strftime("%B %d, %Y")
        
        # Select templates based on experience level
        if is_fresher:
            intro = random.choice(self.fresher_intro_templates)
            body = random.choice(self.fresher_body_templates)
            closing = random.choice(self.fresher_closing_templates)
        else:
            intro = random.choice(self.experienced_intro_templates)
            body = random.choice(self.experienced_body_templates)
            closing = random.choice(self.experienced_closing_templates)
        
        company_paragraph = random.choice(self.company_paragraph_templates)
        
        # Fill in the templates
        intro = intro.format(
            job_title=job_title,
            company_name=company_name,
            background=background,
            experience_years=experience_years,
            education=education
        )
        
        if is_fresher:
            body = body.format(
                skills_list=skills_list,
                project_type=project_type,
                achievement_1=achievement_1,
                achievement_2=achievement_2,
                relevant_skill=relevant_skill
            )
        else:
            body = body.format(
                skills_list=skills_list,
                previous_company=previous_company,
                achievement_1=achievement_1,
                achievement_2=achievement_2,
                relevant_skill=relevant_skill
            )
        
        company_paragraph = company_paragraph.format(
            company_name=company_name,
            company_value=company_value,
            company_project=company_project,
            future_goal=future_goal
        )
        
        closing = closing.format(company_name=company_name)
        
        # Assemble the cover letter
        cover_letter = f"""
{current_date}

{candidate_name}
{candidate_location}
{candidate_phone}
{candidate_email}

Hiring Manager
{company_name}

Subject: Application for {job_title} Position

Dear Hiring Manager,

{intro}

{body}

{company_paragraph}

{closing}

Sincerely,
{candidate_name}
"""
        
        return cover_letter.strip()

# Job Search Engine Class
class JobSearchEngine:
    def __init__(self):
        # Major cities in India for data science jobs
        self.india_cities = [
            "Bangalore", "Bengaluru", "Mumbai", "Delhi", "Hyderabad", 
            "Chennai", "Pune", "Gurgaon", "Noida", "Kolkata"
        ]
        
        # Data science related job titles
        self.data_science_titles = [
            "Data Scientist", "Machine Learning Engineer", "Data Analyst",
            "AI Engineer", "Data Engineer", "Business Intelligence Analyst",
            "Research Scientist", "ML Ops Engineer", "Analytics Manager"
        ]
    
    def search_jobs(self, keywords=None, location=None, experience=None, job_type=None, limit=20):
        """
        Search for data science jobs based on criteria
        """
        import random
        
        # Default keywords if none provided
        if not keywords:
            keywords = ["data science", "machine learning", "python"]
        
        # Default location if none provided
        if not location:
            location = random.choice(self.india_cities)
        
        # Generate random job listings
        jobs = []
        
        for i in range(limit):
            # Generate a random job
            job_id = f"JOB-{random.randint(100000, 999999)}"
            title = random.choice(self.data_science_titles)
            company = f"{self._generate_company_name()} {random.choice(['Technologies', 'Solutions', 'Inc', 'Ltd', 'Analytics'])}"
            job_location = location if location else random.choice(self.india_cities)
            
            # Generate experience range
            if experience:
                min_exp = max(0, int(experience) - 2)
                max_exp = int(experience) + 3
            else:
                min_exp = random.randint(0, 5)
                max_exp = min_exp + random.randint(2, 5)
            
            experience_range = f"{min_exp} - {max_exp} years"
            
            # Generate salary range
            min_salary = random.randint(5, 40)
            max_salary = min_salary + random.randint(5, 40)
            salary = f"₹{min_salary} - {max_salary} LPA"
            
            # Use provided job type or random
            job_type_value = job_type if job_type else random.choice(["Full-time", "Contract", "Part-time", "Remote"])
            
            # Generate posting date
            days_ago = random.randint(1, 30)
            posting_date = (datetime.now().strftime("%Y-%m-%d"))
            
            # Generate short description
            description = self._generate_short_description(title, keywords)
            
            # Generate skills list
            skills = self._generate_skills_list(keywords)
            
            job = {
                "id": job_id,
                "title": title,
                "company": company,
                "location": job_location,
                "experience": experience_range,
                "salary": salary,
                "job_type": job_type_value,
                "posted_date": posting_date,
                "description": description,
                "skills": skills,
                "application_url": f"https://example.com/jobs/{job_id}/apply"
            }
            
            jobs.append(job)
        
        return jobs
    
    def get_job_details(self, job_id):
        """
        Get detailed information about a specific job
        """
        import random
        
        # Generate random job details
        job_details = {
            "id": job_id,
            "title": random.choice(self.data_science_titles),
            "company": f"{self._generate_company_name()} {random.choice(['Technologies', 'Solutions', 'Inc', 'Ltd', 'Analytics'])}",
            "location": random.choice(self.india_cities),
            "salary": f"₹{random.randint(5, 40)} - {random.randint(41, 80)} LPA",
            "experience": f"{random.randint(0, 5)} - {random.randint(6, 15)} years",
            "job_type": random.choice(["Full-time", "Contract", "Part-time", "Remote"]),
            "posted_date": datetime.now().strftime("%Y-%m-%d"),
            "description": self._generate_job_description(),
            "requirements": self._generate_job_requirements(),
            "benefits": self._generate_job_benefits(),
            "application_url": f"https://example.com/jobs/{job_id}/apply",
            "company_url": "https://example.com/company",
            "contact_email": "hr@example.com"
        }
        
        return job_details
    
    def apply_to_job(self, job_id, resume_data, cover_letter=None):
        """
        Apply to a specific job
        """
        import random
        
        # Generate random application result
        job_details = self.get_job_details(job_id)
        
        # Simulate success with high probability
        success = random.random() < 0.95
        
        result = {
            "job_id": job_id,
            "company": job_details["company"],
            "title": job_details["title"],
            "applied_date": datetime.now().strftime("%Y-%m-%d"),
            "status": "success" if success else "failed",
            "message": "Application submitted successfully" if success else "Failed to submit application",
            "reference_id": f"REF-{random.randint(100000, 999999)}" if success else None
        }
        
        return result
    
    def auto_apply_to_jobs(self, resume_data, generate_cover_letter_func, location=None, 
                          experience=None, job_type=None, min_match_score=70, max_applications=1000):
        """
        Automatically apply to jobs that match the candidate's resume
        """
        import random
        import time
        
        # Find matching jobs
        matching_jobs = self.find_matching_jobs(resume_data, location, experience, job_type, limit=max_applications*2)
        
        # Filter jobs by match score
        qualified_jobs = [job for job in matching_jobs if job["match_score"] >= min_match_score]
        
        # Limit to max_applications
        jobs_to_apply = qualified_jobs[:max_applications]
        
        # Apply to jobs
        application_results = []
        for job in jobs_to_apply:
            # Get job details
            job_details = self.get_job_details(job["id"])
            
            # Generate cover letter
            cover_letter = generate_cover_letter_func(resume_data, job_details)
            
            # Apply to job
            result = self.apply_to_job(job["id"], resume_data, cover_letter)
            application_results.append(result)
            
            # Add some delay between applications
            time.sleep(0.1)  # Reduced for demo purposes
        
        # Calculate statistics
        successful_applications = sum(1 for result in application_results if result["status"] == "success")
        
        return {
            "total_matching_jobs": len(matching_jobs),
            "qualified_jobs": len(qualified_jobs),
            "applications_submitted": len(application_results),
            "successful_applications": successful_applications,
            "application_success_rate": successful_applications / len(application_results) if application_results else 0,
            "applied_jobs": [{"job_id": result["job_id"], "company": result["company"], "title": result["title"], 
                             "status": result["status"]} for result in application_results]
        }
    
    def find_matching_jobs(self, resume_data, location=None, experience=None, job_type=None, limit=50):
        """
        Find jobs that match the candidate's resume
        """
        import random
        
        # Extract keywords from resume
        keywords = list(resume_data.get("keyword_matches", {}).keys())
        
        # Search for jobs
        jobs = self.search_jobs(keywords, location, experience, job_type, limit)
        
        # Calculate match score for each job
        matching_jobs = []
        for job in jobs:
            # Calculate a random match score between 60 and 95
            match_score = random.randint(60, 95)
            job["match_score"] = match_score
            matching_jobs.append(job)
        
        # Sort by match score (descending)
        matching_jobs.sort(key=lambda x: x["match_score"], reverse=True)
        
        return matching_jobs
    
    def _generate_company_name(self):
        """Generate a random company name"""
        import random
        
        prefixes = ["Tech", "Data", "AI", "Quantum", "Neural", "Cloud", "Digital", "Cyber", "Smart", "Future"]
        suffixes = ["Systems", "Analytics", "Networks", "Labs", "Works", "Minds", "Soft", "Tech", "Logic", "Nexus"]
        
        return f"{random.choice(prefixes)}{random.choice(suffixes)}"
    
    def _generate_short_description(self, title, keywords):
        """Generate a short job description"""
        import random
        
        templates = [
            "We are looking for a talented {title} to join our team. The ideal candidate will have experience with {skills}.",
            "Exciting opportunity for a {title} with expertise in {skills} to work on cutting-edge projects.",
            "Join our growing team as a {title}. You will work on {skills} to solve complex business problems."
        ]
        
        template = random.choice(templates)
        skills = ", ".join(random.sample(keywords, min(3, len(keywords))))
        
        return template.format(title=title, skills=skills)
    
    def _generate_skills_list(self, keywords):
        """Generate a list of required skills"""
        import random
        
        # Include provided keywords
        skills = list(keywords)
        
        # Add some random technical skills
        technical_skills = ["Python", "R", "SQL", "Spark", "Hadoop", "TensorFlow", "PyTorch", 
                           "scikit-learn", "Pandas", "NumPy", "Tableau", "Power BI", "AWS", "Azure"]
        
        skills.extend(random.sample(technical_skills, random.randint(3, 6)))
        
        # Add some random soft skills
        soft_skills = ["Communication", "Teamwork", "Problem-solving", "Critical thinking", 
                      "Presentation", "Project management", "Time management"]
        
        skills.extend(random.sample(soft_skills, random.randint(1, 3)))
        
        # Remove duplicates and return
        return list(set(skills))
    
    def _generate_job_description(self):
        """Generate a detailed job description"""
        import random
        
        intro_templates = [
            "We are seeking a talented and motivated professional to join our data science team.",
            "Our client is looking for an experienced data professional to strengthen their analytics capabilities.",
            "An exciting opportunity has arisen for a data specialist to join our innovative team."
        ]
        
        responsibility_templates = [
            "Develop and implement machine learning models to solve complex business problems.",
            "Analyze large datasets to extract actionable insights and support decision-making.",
            "Collaborate with cross-functional teams to understand business requirements and deliver data-driven solutions.",
            "Build and maintain data pipelines for efficient data processing and analysis.",
            "Create visualizations and dashboards to communicate findings to stakeholders.",
            "Conduct statistical analysis and hypothesis testing to validate assumptions.",
            "Implement and optimize algorithms for improved performance and accuracy.",
            "Participate in the full data science lifecycle from problem definition to model deployment."
        ]
        
        intro = random.choice(intro_templates)
        responsibilities = random.sample(responsibility_templates, random.randint(4, 6))
        
        description = f"{intro}\n\nKey Responsibilities:\n" + "\n".join([f"- {resp}" for resp in responsibilities])
        
        return description
    
    def _generate_job_requirements(self):
        """Generate job requirements"""
        import random
        
        education_templates = [
            "Bachelor's degree in Computer Science, Statistics, Mathematics, or related field. Master's degree preferred.",
            "Master's or PhD in Computer Science, Data Science, or related quantitative field.",
            "Degree in Computer Science, Engineering, Statistics, or equivalent practical experience."
        ]
        
        experience_templates = [
            "Minimum {min_years}-{max_years} years of experience in data science, machine learning, or related field.",
            "{min_years}+ years of hands-on experience with data analysis and machine learning projects.",
            "At least {min_years} years of professional experience in data science or analytics roles."
        ]
        
        technical_requirements = [
            "Proficiency in Python and its data science libraries (Pandas, NumPy, scikit-learn).",
            "Experience with machine learning frameworks such as TensorFlow, PyTorch, or Keras.",
            "Strong SQL skills and experience working with relational databases.",
            "Familiarity with big data technologies (Hadoop, Spark, Hive).",
            "Knowledge of data visualization tools (Tableau, Power BI, or similar).",
            "Experience with cloud platforms (AWS, Azure, or GCP).",
            "Understanding of statistical concepts and methods.",
            "Experience with version control systems (Git)."
        ]
        
        soft_requirements = [
            "Excellent problem-solving and analytical thinking abilities.",
            "Strong communication skills with the ability to explain complex concepts to non-technical stakeholders.",
            "Ability to work collaboratively in cross-functional teams.",
            "Self-motivated with a strong attention to detail."
        ]
        
        min_years = random.randint(1, 5)
        max_years = min_years + random.randint(2, 5)
        
        education = random.choice(education_templates)
        experience = random.choice(experience_templates).format(min_years=min_years, max_years=max_years)
        tech_reqs = random.sample(technical_requirements, random.randint(3, 5))
        soft_reqs = random.sample(soft_requirements, random.randint(2, 3))
        
        requirements = f"Requirements:\n\n- {education}\n- {experience}\n" + \
                      "\n".join([f"- {req}" for req in tech_reqs + soft_reqs])
        
        return requirements
    
    def _generate_job_benefits(self):
        """Generate job benefits"""
        import random
        
        benefits_list = [
            "Competitive salary package",
            "Performance-based bonuses",
            "Health insurance for you and your family",
            "Flexible working hours",
            "Remote work options",
            "Professional development opportunities",
            "Training and conference attendance",
            "Modern office with recreational facilities",
            "Regular team outings and events",
            "Casual dress code"
        ]
        
        selected_benefits = random.sample(benefits_list, random.randint(4, 7))
        
        benefits = "Benefits:\n\n" + "\n".join([f"- {benefit}" for benefit in selected_benefits])
        
        return benefits

# Initialize components
resume_parser = ResumeParser()
ats_scorer = ATSScorer()
cover_letter_generator = CoverLetterGenerator()
job_search_engine = JobSearchEngine()

# Session state initialization
if 'resume_data' not in st.session_state:
    st.session_state.resume_data = None
if 'job_data' not in st.session_state:
    st.session_state.job_data = None
if 'ats_score' not in st.session_state:
    st.session_state.ats_score = None
if 'cover_letter' not in st.session_state:
    st.session_state.cover_letter = None
if 'jobs' not in st.session_state:
    st.session_state.jobs = None
if 'job_details' not in st.session_state:
    st.session_state.job_details = None
if 'application_result' not in st.session_state:
    st.session_state.application_result = None
if 'auto_apply_results' not in st.session_state:
    st.session_state.auto_apply_results = None

# Sidebar
st.sidebar.title("Data Science Job Bot")
st.sidebar.image("https://img.icons8.com/color/96/000000/data-science.png", width=100)

# Navigation
page = st.sidebar.radio("Navigation", ["Home", "Resume Upload & Analysis", "Job Search", "Cover Letter Generator", "Job Application", "Auto Apply"])

# Home page
if page == "Home":
    st.title("Data Science Job Application Assistant")
    st.markdown("""
    Welcome to the Data Science Job Application Assistant! This application helps you find and apply to data science jobs across India.
    
    ### Features:
    - **Resume Upload & Analysis**: Upload your resume and get it parsed and analyzed
    - **ATS Score**: Check how well your resume matches job descriptions
    - **Job Search**: Find data science jobs across India
    - **Cover Letter Generator**: Generate customized cover letters for job applications
    - **Job Application**: Apply to specific jobs with your resume and cover letter
    - **Auto Apply**: Automatically apply to multiple jobs that match your profile
    
    ### How to use:
    1. Start by uploading your resume in the "Resume Upload & Analysis" section
    2. Search for jobs in the "Job Search" section
    3. Generate cover letters for specific jobs
    4. Apply to jobs individually or use the auto-apply feature
    
    This application can help you apply to up to 1000 data science jobs throughout India!
    """)
    
    st.info("Note: This is a demonstration application. In a production environment, it would connect to real job portals and APIs.")

# Resume Upload & Analysis page
elif page == "Resume Upload & Analysis":
    st.title("Resume Upload & Analysis")
    
    # Resume upload
    st.header("Upload Resume")
    uploaded_file = st.file_uploader("Upload your resume (PDF, DOC, DOCX)", type=["pdf", "doc", "docx"])
    
    if uploaded_file is not None:
        # Save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            file_path = tmp_file.name
        
        # Parse resume
        if st.button("Parse Resume"):
            with st.spinner("Parsing resume..."):
                try:
                    resume_data = resume_parser.parse_resume(file_path)
                    st.session_state.resume_data = resume_data
                    st.success("Resume parsed successfully!")
                    
                    # Display parsed data
                    st.subheader("Parsed Resume Data")
                    
                    # Contact info
                    contact_info = resume_data.get("contact_info", {})
                    st.write(f"**Name:** {contact_info.get('name', 'N/A')}")
                    st.write(f"**Email:** {contact_info.get('email', 'N/A')}")
                    st.write(f"**Phone:** {contact_info.get('phone', 'N/A')}")
                    st.write(f"**Location:** {contact_info.get('location', 'N/A')}")
                    
                    # Education and Experience
                    st.write(f"**Education:** {resume_data.get('education', 'N/A')}")
                    st.write(f"**Experience:** {resume_data.get('experience', 'N/A')}")
                    
                    # Skills
                    st.subheader("Skills")
                    skills = resume_data.get("skills", [])
                    if skills:
                        st.write(", ".join(skills))
                    else:
                        st.write("No skills extracted")
                    
                    # Keyword matches
                    st.subheader("Keyword Matches")
                    keyword_matches = resume_data.get("keyword_matches", {})
                    if keyword_matches:
                        for keyword, count in keyword_matches.items():
                            st.write(f"- {keyword}: {count}")
                    else:
                        st.write("No keyword matches found")
                    
                except Exception as e:
                    st.error(f"Error parsing resume: {str(e)}")
    
    # ATS Score Analysis
    st.header("ATS Score Analysis")
    
    if st.session_state.resume_data is not None:
        job_description = st.text_area("Enter Job Description", height=200)
        
        if st.button("Analyze ATS Score"):
            if job_description:
                with st.spinner("Analyzing resume against job description..."):
                    ats_score = ats_scorer.score_resume_for_job(
                        st.session_state.resume_data.get("resume_text", ""),
                        job_description
                    )
                    st.session_state.ats_score = ats_score
                    
                    # Display ATS score results
                    st.subheader("ATS Score Results")
                    
                    # Overall match percentage
                    match_percentage = ats_score.get("match_percentage", 0)
                    st.metric("Overall Match Percentage", f"{match_percentage:.1f}%")
                    
                    # Section scores
                    st.subheader("Section Scores")
                    section_scores = ats_score.get("section_scores", {})
                    cols = st.columns(len(section_scores))
                    for i, (section, score) in enumerate(section_scores.items()):
                        if section != "overall":  # Skip overall since we already displayed it
                            cols[i].metric(section.capitalize(), f"{score:.1f}%")
                    
                    # Keyword matches
                    st.subheader("Keyword Matches")
                    keyword_matches = ats_score.get("keyword_matches", {})
                    if keyword_matches:
                        for keyword, count in keyword_matches.items():
                            st.write(f"- {keyword}: {count}")
                    else:
                        st.write("No keyword matches found")
                    
                    # Missing keywords
                    st.subheader("Missing Keywords")
                    missing_keywords = ats_score.get("missing_keywords", [])
                    if missing_keywords:
                        for keyword in missing_keywords:
                            st.write(f"- {keyword}")
                    else:
                        st.write("No missing keywords")
                    
                    # Improvement suggestions
                    st.subheader("Improvement Suggestions")
                    suggestions = ats_score.get("improvement_suggestions", [])
                    if suggestions:
                        for suggestion in suggestions:
                            st.write(f"- {suggestion}")
                    else:
                        st.write("No improvement suggestions")
            else:
                st.warning("Please enter a job description")
    else:
        st.warning("Please upload and parse a resume first")

# Job Search page
elif page == "Job Search":
    st.title("Job Search")
    
    # Search form
    st.header("Search for Data Science Jobs")
    
    col1, col2 = st.columns(2)
    with col1:
        keywords = st.text_input("Keywords (comma separated)", "python, machine learning, data science")
        location = st.text_input("Location", "Bangalore")
    
    with col2:
        experience = st.text_input("Experience (years)", "3")
        job_type = st.selectbox("Job Type", ["Full-time", "Contract", "Part-time", "Remote"])
    
    limit = st.slider("Number of Jobs", 5, 100, 20)
    
    if st.button("Search Jobs"):
        with st.spinner("Searching for jobs..."):
            keywords_list = [k.strip() for k in keywords.split(',') if k.strip()]
            jobs = job_search_engine.search_jobs(
                keywords=keywords_list,
                location=location,
                experience=experience,
                job_type=job_type,
                limit=limit
            )
            st.session_state.jobs = jobs
            
            # Display search results
            st.subheader(f"Found {len(jobs)} Jobs")
            
            for job in jobs:
                with st.expander(f"{job['title']} at {job['company']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Location:** {job['location']}")
                        st.write(f"**Experience:** {job['experience']}")
                    with col2:
                        st.write(f"**Salary:** {job['salary']}")
                        st.write(f"**Job Type:** {job['job_type']}")
                    
                    st.write(f"**Description:** {job['description']}")
                    
                    st.write("**Skills:**")
                    skills = job.get('skills', [])
                    if skills:
                        st.write(", ".join(skills))
                    
                    # Buttons for actions
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button(f"View Details #{job['id']}", key=f"details_{job['id']}"):
                            st.session_state.job_details = job_search_engine.get_job_details(job['id'])
                            st.session_state.job_data = st.session_state.job_details
                    with col2:
                        if st.button(f"Generate Cover Letter #{job['id']}", key=f"cover_{job['id']}"):
                            if st.session_state.resume_data is not None:
                                job_details = job_search_engine.get_job_details(job['id'])
                                st.session_state.job_data = job_details
                                st.session_state.cover_letter = cover_letter_generator.generate_cover_letter(
                                    st.session_state.resume_data,
                                    job_details
                                )
                            else:
                                st.warning("Please upload and parse a resume first")
                    with col3:
                        if st.button(f"Apply #{job['id']}", key=f"apply_{job['id']}"):
                            if st.session_state.resume_data is not None:
                                job_details = job_search_engine.get_job_details(job['id'])
                                cover_letter = None
                                if st.session_state.cover_letter is not None:
                                    cover_letter = st.session_state.cover_letter
                                st.session_state.application_result = job_search_engine.apply_to_job(
                                    job['id'],
                                    st.session_state.resume_data,
                                    cover_letter
                                )
                            else:
                                st.warning("Please upload and parse a resume first")
    
    # Display job details if available
    if st.session_state.job_details is not None:
        st.header("Job Details")
        job = st.session_state.job_details
        
        st.subheader(f"{job['title']} at {job['company']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Location:** {job['location']}")
            st.write(f"**Experience:** {job['experience']}")
            st.write(f"**Job Type:** {job['job_type']}")
        with col2:
            st.write(f"**Salary:** {job['salary']}")
            st.write(f"**Posted Date:** {job['posted_date']}")
            st.write(f"**Contact:** {job['contact_email']}")
        
        st.subheader("Description")
        st.write(job['description'])
        
        st.subheader("Requirements")
        st.write(job['requirements'])
        
        st.subheader("Benefits")
        st.write(job['benefits'])
        
        # Buttons for actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate Cover Letter", key="gen_cover_details"):
                if st.session_state.resume_data is not None:
                    st.session_state.cover_letter = cover_letter_generator.generate_cover_letter(
                        st.session_state.resume_data,
                        job
                    )
                else:
                    st.warning("Please upload and parse a resume first")
        with col2:
            if st.button("Apply to Job", key="apply_details"):
                if st.session_state.resume_data is not None:
                    cover_letter = None
                    if st.session_state.cover_letter is not None:
                        cover_letter = st.session_state.cover_letter
                    st.session_state.application_result = job_search_engine.apply_to_job(
                        job['id'],
                        st.session_state.resume_data,
                        cover_letter
                    )
                else:
                    st.warning("Please upload and parse a resume first")

# Cover Letter Generator page
elif page == "Cover Letter Generator":
    st.title("Cover Letter Generator")
    
    if st.session_state.resume_data is None:
        st.warning("Please upload and parse a resume first in the 'Resume Upload & Analysis' section")
    else:
        # Job data input
        st.header("Job Information")
        
        col1, col2 = st.columns(2)
        with col1:
            job_title = st.text_input("Job Title", "Data Scientist")
            company_name = st.text_input("Company Name", "TechSolutions Inc")
        
        with col2:
            job_location = st.text_input("Job Location", "Bangalore, India")
            job_type = st.selectbox("Job Type", ["Full-time", "Contract", "Part-time", "Remote"], key="cl_job_type")
        
        job_description = st.text_area("Job Description", height=200)
        
        # Use existing job data if available
        use_existing = False
        if st.session_state.job_data is not None:
            use_existing = st.checkbox("Use existing job data")
            if use_existing:
                job_data = st.session_state.job_data
                st.info(f"Using data for {job_data.get('title')} at {job_data.get('company')}")
        
        if st.button("Generate Cover Letter"):
            with st.spinner("Generating cover letter..."):
                if use_existing and st.session_state.job_data is not None:
                    job_data = st.session_state.job_data
                else:
                    job_data = {
                        "title": job_title,
                        "company": company_name,
                        "location": job_location,
                        "job_type": job_type,
                        "description": job_description
                    }
                
                cover_letter = cover_letter_generator.generate_cover_letter(
                    st.session_state.resume_data,
                    job_data
                )
                st.session_state.cover_letter = cover_letter
                st.session_state.job_data = job_data
        
        # Display generated cover letter
        if st.session_state.cover_letter is not None:
            st.header("Generated Cover Letter")
            st.text_area("Cover Letter", st.session_state.cover_letter, height=400)
            
            # Download button
            if st.download_button(
                label="Download Cover Letter",
                data=st.session_state.cover_letter,
                file_name="cover_letter.txt",
                mime="text/plain"
            ):
                st.success("Cover letter downloaded successfully!")

# Job Application page
elif page == "Job Application":
    st.title("Job Application")
    
    if st.session_state.resume_data is None:
        st.warning("Please upload and parse a resume first in the 'Resume Upload & Analysis' section")
    else:
        # Job selection
        st.header("Apply to Job")
        
        # Option to use existing job data
        use_existing = False
        if st.session_state.job_data is not None:
            use_existing = st.checkbox("Use existing job data")
            if use_existing:
                job_data = st.session_state.job_data
                st.info(f"Using data for {job_data.get('title')} at {job_data.get('company')}")
                job_id = job_data.get('id')
        
        if not use_existing:
            job_id = st.text_input("Job ID")
        
        # Cover letter
        st.subheader("Cover Letter")
        
        use_generated = False
        if st.session_state.cover_letter is not None:
            use_generated = st.checkbox("Use generated cover letter")
            if use_generated:
                cover_letter = st.session_state.cover_letter
                st.info("Using previously generated cover letter")
        
        if not use_generated:
            cover_letter = st.text_area("Cover Letter (optional)", height=200)
        
        if st.button("Apply to Job"):
            if job_id:
                with st.spinner("Applying to job..."):
                    application_result = job_search_engine.apply_to_job(
                        job_id,
                        st.session_state.resume_data,
                        cover_letter if cover_letter else None
                    )
                    st.session_state.application_result = application_result
            else:
                st.warning("Please enter a Job ID")
        
        # Display application result
        if st.session_state.application_result is not None:
            st.header("Application Result")
            
            result = st.session_state.application_result
            
            if result.get('status') == 'success':
                st.success(result.get('message', 'Application submitted successfully'))
            else:
                st.error(result.get('message', 'Failed to submit application'))
            
            st.write(f"**Job ID:** {result.get('job_id')}")
            st.write(f"**Company:** {result.get('company')}")
            st.write(f"**Title:** {result.get('title')}")
            st.write(f"**Applied Date:** {result.get('applied_date')}")
            
            if result.get('reference_id'):
                st.write(f"**Reference ID:** {result.get('reference_id')}")

# Auto Apply page
elif page == "Auto Apply":
    st.title("Auto Apply to Multiple Jobs")
    
    if st.session_state.resume_data is None:
        st.warning("Please upload and parse a resume first in the 'Resume Upload & Analysis' section")
    else:
        st.header("Auto Apply Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            location = st.text_input("Preferred Location", "Bangalore")
            experience = st.text_input("Experience (years)", "3")
        
        with col2:
            job_type = st.selectbox("Job Type", ["Full-time", "Contract", "Part-time", "Remote"], key="auto_job_type")
            min_match = st.slider("Minimum Match Score (%)", 50, 90, 70, 5)
        
        max_apps = st.slider("Maximum Applications", 10, 1000, 100, 10)
        
        if st.button("Start Auto Apply"):
            with st.spinner(f"Searching and applying to up to {max_apps} jobs..."):
                # Create a function to generate cover letters for each job
                def generate_cover_letter_for_job(resume_data, job_details):
                    return cover_letter_generator.generate_cover_letter(resume_data, job_details)
                
                # Run auto-apply process
                auto_apply_results = job_search_engine.auto_apply_to_jobs(
                    resume_data=st.session_state.resume_data,
                    generate_cover_letter_func=generate_cover_letter_for_job,
                    location=location,
                    experience=experience,
                    job_type=job_type,
                    min_match_score=min_match,
                    max_applications=max_apps
                )
                
                st.session_state.auto_apply_results = auto_apply_results
        
        # Display auto apply results
        if st.session_state.auto_apply_results is not None:
            st.header("Auto Apply Results")
            
            results = st.session_state.auto_apply_results
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Matching Jobs", results.get('total_matching_jobs', 0))
            col2.metric("Qualified Jobs", results.get('qualified_jobs', 0))
            col3.metric("Applications Submitted", results.get('applications_submitted', 0))
            col4.metric("Successful Applications", results.get('successful_applications', 0))
            
            # Success rate
            success_rate = results.get('application_success_rate', 0) * 100
            st.metric("Application Success Rate", f"{success_rate:.1f}%")
            
            # Applied jobs
            st.subheader("Applied Jobs")
            
            applied_jobs = results.get('applied_jobs', [])
            if applied_jobs:
                # Create a DataFrame for better display
                df = pd.DataFrame(applied_jobs)
                st.dataframe(df)
                
                # Download button for application results
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Application Results",
                    data=csv,
                    file_name="application_results.csv",
                    mime="text/csv"
                )
            else:
                st.write("No jobs applied to")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("This application can apply to up to 1000 data science jobs across India!")
