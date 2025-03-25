import streamlit as st
import os
import sqlite3
from datetime import datetime
import PyPDF2
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import platform
import psutil
import GPUtil
import plotly.graph_objs as go
import time

# Hugging Face dependencies
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

@dataclass
class ProjectMetadata:
    name: str
    description: str = ""
    type: str = "Other"
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    last_updated: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

@dataclass
class PaperMetadata:
    filename: str
    title: str
    path: str
    preview: str = ""
    author: str = ""
    year: str = ""
    doi: str = ""

@dataclass
class Project:
    metadata: ProjectMetadata
    paper: Optional[PaperMetadata] = None
    tasks: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

class ProjectDatabase:
    def __init__(self, db_path='projects.db'):
        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                description TEXT,
                type TEXT,
                created_at TEXT,
                last_updated TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                project_name TEXT,
                filename TEXT,
                title TEXT,
                path TEXT,
                preview TEXT,
                author TEXT,
                year TEXT,
                doi TEXT,
                FOREIGN KEY(project_name) REFERENCES projects(name)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                project_name TEXT,
                task TEXT,
                FOREIGN KEY(project_name) REFERENCES projects(name)
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS dependencies (
                project_name TEXT,
                dependency TEXT,
                FOREIGN KEY(project_name) REFERENCES projects(name)
            )
        ''')
        self.conn.commit()

    def save_project(self, project: Project):
        cursor = self.conn.cursor()
        
        # Insert/Update Project Metadata
        cursor.execute('''
            INSERT OR REPLACE INTO projects 
            (name, description, type, created_at, last_updated) 
            VALUES (?, ?, ?, ?, ?)
        ''', (
            project.metadata.name, 
            project.metadata.description, 
            project.metadata.type, 
            project.metadata.created_at, 
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        # Save Paper if exists
        if project.paper:
            cursor.execute('''
                INSERT OR REPLACE INTO papers 
                (project_name, filename, title, path, preview, author, year, doi) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                project.metadata.name,
                project.paper.filename,
                project.paper.title,
                project.paper.path,
                project.paper.preview,
                project.paper.author,
                project.paper.year,
                project.paper.doi
            ))

        # Save Tasks
        cursor.execute('DELETE FROM tasks WHERE project_name = ?', (project.metadata.name,))
        for task in project.tasks:
            cursor.execute('INSERT INTO tasks (project_name, task) VALUES (?, ?)', 
                           (project.metadata.name, task))

        # Save Dependencies
        cursor.execute('DELETE FROM dependencies WHERE project_name = ?', (project.metadata.name,))
        for dep in project.dependencies:
            cursor.execute('INSERT INTO dependencies (project_name, dependency) VALUES (?, ?)', 
                           (project.metadata.name, dep))

        self.conn.commit()

    def get_projects(self) -> List[Project]:
        cursor = self.conn.cursor()
        
        # Fetch projects
        cursor.execute('SELECT * FROM projects')
        project_rows = cursor.fetchall()
        
        projects = []
        for row in project_rows:
            # Fetch paper
            cursor.execute('SELECT * FROM papers WHERE project_name = ?', (row[1],))
            paper_row = cursor.fetchone()
            
            # Fetch tasks
            cursor.execute('SELECT task FROM tasks WHERE project_name = ?', (row[1],))
            tasks = [task[0] for task in cursor.fetchall()]
            
            # Fetch dependencies
            cursor.execute('SELECT dependency FROM dependencies WHERE project_name = ?', (row[1],))
            dependencies = [dep[0] for dep in cursor.fetchall()]
            
            # Create Project object
            metadata = ProjectMetadata(
                name=row[1],
                description=row[2],
                type=row[3],
                created_at=row[4],
                last_updated=row[5]
            )
            
            paper = None
            if paper_row:
                paper = PaperMetadata(
                    filename=paper_row[1],
                    title=paper_row[2],
                    path=paper_row[3],
                    preview=paper_row[4],
                    author=paper_row[5],
                    year=paper_row[6],
                    doi=paper_row[7]
                )
            
            project = Project(
                metadata=metadata,
                paper=paper,
                tasks=tasks,
                dependencies=dependencies
            )
            projects.append(project)
        
        return projects

    def delete_project(self, project_name: str):
        cursor = self.conn.cursor()
        cursor.execute('DELETE FROM projects WHERE name = ?', (project_name,))
        cursor.execute('DELETE FROM papers WHERE project_name = ?', (project_name,))
        cursor.execute('DELETE FROM tasks WHERE project_name = ?', (project_name,))
        cursor.execute('DELETE FROM dependencies WHERE project_name = ?', (project_name,))
        self.conn.commit()

class LLMTaskGenerator:
    def __init__(self):
        try:
            # Use a small, non-gated model
            model_name = "facebook/opt-125m"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Create a text generation pipeline
            self.generator = pipeline(
                'text-generation', 
                model=self.model, 
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
            st.error(f"Error loading language model: {e}")
            self.generator = None

    def generate_tasks_and_dependencies(self, paper_text: str) -> Dict[str, List[str]]:
        if not self.generator:
            return {
                'tasks': ['Manual task review needed'],
                'dependencies': ['Manual dependency review needed']
            }

        try:
            # Truncate input to first 1000 characters to avoid context limitations
            truncated_text = paper_text[:1000]

            # Generate tasks prompt
            tasks_prompt = f"List step-by-step tasks to reproduce the research in this paper: {truncated_text}"
            tasks_generation = self.generator(
                tasks_prompt, 
                max_length=500, 
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7
            )[0]['generated_text']

            # Generate dependencies prompt
            deps_prompt = f"List Python libraries and tools needed to reproduce this research paper: {truncated_text}"
            deps_generation = self.generator(
                deps_prompt, 
                max_length=300, 
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7
            )[0]['generated_text']

            # Post-process generations
            def extract_list_items(text):
                # Extract items starting with number or dash
                import re
                items = re.findall(r'[\d\-•][\s]*([^\n]+)', text)
                # Remove duplicates and strip whitespace
                return list(dict.fromkeys(item.strip() for item in items if item.strip()))

            tasks = extract_list_items(tasks_generation)
            dependencies = extract_list_items(deps_generation)

            # Fallback if no items extracted
            if not tasks:
                tasks = ['Review paper manually for reproduction steps']
            if not dependencies:
                dependencies = ['Review paper manually for dependencies']

            return {
                'tasks': tasks[:5],  # Limit to 5 items
                'dependencies': dependencies[:5]  # Limit to 5 items
            }

        except Exception as e:
            st.error(f"Error generating tasks and dependencies: {e}")
            return {
                'tasks': ['Manual task review needed'],
                'dependencies': ['Manual dependency review needed']
            }

class PDFProcessor:
    @staticmethod
    def extract_metadata(pdf_path: str) -> Optional[PaperMetadata]:
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                metadata = reader.metadata or {}
                
                title = metadata.get('/Title', os.path.basename(pdf_path))
                author = metadata.get('/Author', 'Unknown')
                
                first_page = reader.pages[0].extract_text()[:500] if reader.pages else ""
                
                return PaperMetadata(
                    filename=os.path.basename(pdf_path),
                    title=title,
                    path=pdf_path,
                    preview=first_page,
                    author=author,
                    year=metadata.get('/CreationDate', '')[:4] if metadata.get('/CreationDate') else ''
                )
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            return None

    @staticmethod
    def extract_text(pdf_path: str) -> str:
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text[:2000]  # Limit to first 2000 characters
        except Exception as e:
            st.error(f"Error extracting PDF text: {e}")
            return ""

    @staticmethod
    def scan_pdf_directory(directory: str) -> List[PaperMetadata]:
        papers = []
        for filename in os.listdir(directory):
            if filename.lower().endswith('.pdf'):
                full_path = os.path.join(directory, filename)
                paper_info = PDFProcessor.extract_metadata(full_path)
                if paper_info:
                    papers.append(paper_info)
        return papers

class SystemMonitor:
    @staticmethod
    def get_system_info():
        # CPU Information
        cpu_info = {
            'Physical Cores': psutil.cpu_count(logical=False),
            'Total Cores': psutil.cpu_count(logical=True),
            'CPU Usage': f"{psutil.cpu_percent()}%"
        }

        # Memory Information
        memory = psutil.virtual_memory()
        memory_info = {
            'Total Memory': f"{memory.total / (1024 ** 3):.2f} GB",
            'Available Memory': f"{memory.available / (1024 ** 3):.2f} GB",
            'Memory Usage': f"{memory.percent}%"
        }

        # GPU Information
        try:
            gpus = GPUtil.getGPUs()
            gpu_info = [{
                'Name': gpu.name,
                'Total Memory': f"{gpu.memoryTotal} MB",
                'Memory Usage': f"{gpu.memoryUtil * 100:.2f}%",
                'GPU Usage': f"{gpu.load * 100:.2f}%"
            } for gpu in gpus] if gpus else [{'Name': 'No GPU detected'}]
        except:
            gpu_info = [{'Name': 'No GPU detected'}]

        # OS Information
        os_info = {
            'OS': platform.system(),
            'Release': platform.release(),
            'Machine': platform.machine()
        }

        return cpu_info, memory_info, gpu_info, os_info

class ResourceMonitorCharts:
    def __init__(self):
        # Initialize session state for chart data if not exists
        if 'cpu_data' not in st.session_state:
            st.session_state.cpu_data = []
        if 'memory_data' not in st.session_state:
            st.session_state.memory_data = []
        if 'gpu_data' not in st.session_state:
            st.session_state.gpu_data = []

    def get_current_resources(self):
        """Collect current system resource utilization."""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_percent = memory.percent

        # GPU data collection
        try:
            gpus = GPUtil.getGPUs()
            gpu_percent = gpus[0].load * 100 if gpus else 0
        except:
            gpu_percent = 0

        return cpu_percent, memory_percent, gpu_percent

    def update_resource_data(self):
        """Update session state with resource data."""
        current_time = time.time()
        cpu_percent, memory_percent, gpu_percent = self.get_current_resources()

        # Maintain only last 50 data points
        st.session_state.cpu_data.append((current_time, cpu_percent))
        st.session_state.memory_data.append((current_time, memory_percent))
        st.session_state.gpu_data.append((current_time, gpu_percent))

        if len(st.session_state.cpu_data) > 50:
            st.session_state.cpu_data.pop(0)
            st.session_state.memory_data.pop(0)
            st.session_state.gpu_data.pop(0)

    def render_resource_charts(self):
        """Create and render resource utilization charts."""
        st.header("🔄 Real-Time System Resources")

        # Create columns for charts
        col1, col2, col3 = st.columns(3)

        # CPU Utilization Chart
        with col1:
            st.subheader("CPU Utilization")
            cpu_chart = st.empty()

        # Memory Utilization Chart  
        with col2:
            st.subheader("Memory Utilization")
            memory_chart = st.empty()

        # GPU Utilization Chart
        with col3:
            st.subheader("GPU Utilization")
            gpu_chart = st.empty()

        # Real-time update loop
        while True:
            # Update resource data
            self.update_resource_data()

            # Prepare data for Plotly charts
            def prepare_line_chart(data, title, yaxis_title):
                if not data:
                    return None
                
                times, values = zip(*data)
                # Adjust times to be relative
                relative_times = [t - times[0] for t in times]

                return go.Figure(data=[
                    go.Scatter(
                        x=relative_times, 
                        y=values, 
                        mode='lines+markers',
                        line=dict(width=3),
                        marker=dict(size=8)
                    )
                ], layout=go.Layout(
                    title=title,
                    xaxis_title='Seconds Ago',
                    yaxis_title=yaxis_title,
                    yaxis=dict(range=[0, 100]),
                    height=300,
                    margin=dict(l=50, r=50, t=50, b=50)
                ))

            # Create and display charts
            cpu_plotly_chart = prepare_line_chart(
                st.session_state.cpu_data, 
                'CPU Usage', 
                'Utilization (%)'
            )
            memory_plotly_chart = prepare_line_chart(
                st.session_state.memory_data, 
                'Memory Usage', 
                'Utilization (%)'
            )
            gpu_plotly_chart = prepare_line_chart(
                st.session_state.gpu_data, 
                'GPU Usage', 
                'Utilization (%)'
            )

            # Update charts
            if cpu_plotly_chart:
                cpu_chart.plotly_chart(cpu_plotly_chart, use_container_width=True)
            if memory_plotly_chart:
                memory_chart.plotly_chart(memory_plotly_chart, use_container_width=True)
            if gpu_plotly_chart:
                gpu_chart.plotly_chart(gpu_plotly_chart, use_container_width=True)

            # Wait before next update
            time.sleep(1)

class ProjectDashboard:
    def __init__(self):
        self.db = ProjectDatabase()
        self.llm_task_generator = LLMTaskGenerator()
        self.paper_directory = "./papers"
        os.makedirs(self.paper_directory, exist_ok=True)
        st.set_page_config(page_title="Advanced Project Dashboard", layout="wide")
        
    def render_projects_page(self):
        st.header("📊 Project Management")
        
        # Project Creation Column
        col1, col2 = st.columns([1, 2])
        
        with col1:
            with st.form("project_creation_form"):
                st.subheader("Create New Project")
                project_name = st.text_input("Project Name", help="Choose a unique project name")
                project_description = st.text_area("Description", help="Provide a brief project overview")
                project_type = st.selectbox(
                    "Project Type", 
                    ["Research", "Software", "Data Science", "Machine Learning", "Other"]
                )
                
                # Paper Selection
                st.subheader("Attach Research Paper")
                papers = PDFProcessor.scan_pdf_directory(self.paper_directory)
                if papers:
                    selected_paper = st.selectbox(
                        "Select Paper", 
                        options=[paper.filename for paper in papers],
                        format_func=lambda x: next(p.title for p in papers if p.filename == x)
                    )
                    selected_paper_metadata = next(p for p in papers if p.filename == selected_paper)
                else:
                    st.warning("No PDFs found in papers directory")
                    selected_paper_metadata = None
                
                submitted = st.form_submit_button("Create Project")
                
                if submitted and project_name and selected_paper_metadata:
                    # Show progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Step 1: Basic Project Creation
                    status_text.text("Step 1: Creating Project...")
                    progress_bar.progress(10)
                    
                    # Step 2: Extract PDF Text
                    status_text.text("Step 2: Extracting Paper Text...")
                    progress_bar.progress(30)
                    pdf_text = PDFProcessor.extract_text(selected_paper_metadata.path)
                    
                    # Step 3: Generate Tasks and Dependencies
                    status_text.text("Step 3: Generating Reproduction Tasks...")
                    progress_bar.progress(50)
                    llm_output = self.llm_task_generator.generate_tasks_and_dependencies(pdf_text)
                    
                    # Step 4: Create Project
                    status_text.text("Step 4: Finalizing Project...")
                    progress_bar.progress(70)
                    
                    new_project = Project(
                        metadata=ProjectMetadata(
                            name=project_name, 
                            description=project_description, 
                            type=project_type
                        ),
                        paper=selected_paper_metadata,
                        tasks=llm_output['tasks'],
                        dependencies=llm_output['dependencies']
                    )
                    
                    try:
                        # Step 5: Save Project
                        status_text.text("Step 5: Saving Project...")
                        progress_bar.progress(90)
                        self.db.save_project(new_project)
                        
                        # Final Step
                        progress_bar.progress(100)
                        status_text.text("Project created successfully!")
                        
                        # Clear progress indicators
                        import time
                        time.sleep(1)
                        progress_bar.empty()
                        status_text.empty()
                        
                        st.success(f"Project '{project_name}' created successfully!")
                    except sqlite3.IntegrityError:
                        st.error("A project with this name already exists. Choose a unique name.")

        with col2:
            st.subheader("Existing Projects")
            projects = self.db.get_projects()
            
            if projects:
                for project in projects:
                    with st.expander(f"{project.metadata.name} ({project.metadata.type})"):
                        st.write(f"**Description:** {project.metadata.description}")
                        st.write(f"**Created:** {project.metadata.created_at}")
                        
                        if project.paper:
                            st.write(f"**Paper:** {project.paper.title}")
                            st.write(f"**Author:** {project.paper.author}")
                        
                        if project.tasks:
                            st.subheader("Tasks")
                            for task in project.tasks:
                                st.write(f"- {task}")
                        
                        if project.dependencies:
                            st.subheader("Dependencies")
                            for dep in project.dependencies:
                                st.write(f"- {dep}")
                        
                        delete_button = st.button(f"Delete {project.metadata.name}")
                        if delete_button:
                            self.db.delete_project(project.metadata.name)
                            st.rerun()  # Note the change from experimental_rerun to rerun
            else:
                st.info("No projects created yet. Use the form to create a new project.")

    def render_hardware_page(self):
        st.header("💻 System Resources")
        
        # Existing static system info
        cpu_info, memory_info, gpu_info, os_info = SystemMonitor.get_system_info()
        
        # Create columns for different sections
        col1, col2, col3 = st.columns(3)
        
        # CPU Information
        with col1:
            st.subheader("CPU")
            for key, value in cpu_info.items():
                st.metric(key, value)
    
        # Memory Information
        with col2:
            st.subheader("Memory")
            for key, value in memory_info.items():
                st.metric(key, value)
    
        # OS Information
        with col3:
            st.subheader("Operating System")
            for key, value in os_info.items():
                st.metric(key, value)
    
        # Real-time Resource Charts
        ResourceMonitorCharts().render_resource_charts()

    def main(self):
        st.title("🚀 Advanced Project Dashboard")
        
        tab1, tab2, tab3 = st.tabs(["Projects", "Paper Repository", "System Resources"])
        
        with tab1:
            self.render_projects_page()
        
        with tab2:
            st.header("📄 Research Paper Repository")
            papers = PDFProcessor.scan_pdf_directory(self.paper_directory)
            
            if papers:
                for paper in papers:
                    with st.expander(f"{paper.title}"):
                        st.write(f"**Filename:** {paper.filename}")
                        st.write(f"**Author:** {paper.author}")
                        st.write(f"**Year:** {paper.year}")
                        st.text_area("Preview", value=paper.preview, height=100)
            else:
                st.warning("No PDFs found in papers directory")
        
        with tab3:
            self.render_hardware_page()

def main():
    dashboard = ProjectDashboard()
    dashboard.main()

if __name__ == "__main__":
    main()