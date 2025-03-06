import os
from typing import List, Type
from crewai import Agent, Crew, Task, LLM, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException


# --- LLM Setup ---
#os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY"  # Replace with your API key
os.environ["GEMINI_API_KEY"] = "AIzaSyBxOYPifKL584yTxybLp8Wb9O09srKZcMQ"
llm = LLM('gemini/gemini-2.0-flash-lite',
                            verbose=True,
                            temperature=0.7)
# --- Tools ---

class CodeWriterToolInput(BaseModel):
    filename: str = Field(..., description="The name of the file to write to.")
    code: str = Field(..., description="The code to write to the file.")

class CodeWriterTool(BaseTool):
    name: str = "code_writer"
    description: str = "Writes code to a specified file.  Overwrites existing files."
    args_schema: Type[BaseModel] = CodeWriterToolInput

    def _run(self, filename: str, code: str) -> str:
        try:
            directory = os.path.dirname(filename)

            # Create the directory if it doesn't exist
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            with open(filename, "w") as f:
                f.write(code)
            return f"Code written to {filename} successfully."
        except Exception as e:
            return f"Error writing to {filename}: {e}"

class CodeReaderToolInput(BaseModel):
    filename: str = Field(..., description="The name of the file to read from.")

class CodeReaderTool(BaseTool):
    name: str = "code_reader"
    description:str = "Reads code from a specified file."
    args_schema: Type[BaseModel] = CodeReaderToolInput

    def _run(self, filename: str) -> str:
        try:
            with open(filename, "r") as f:
                code = f.read()
            return code
        except FileNotFoundError:
            return f"File not found: {filename}"
        except Exception as e:
            return f"Error reading from {filename}: {e}"

class WebSearchToolInput(BaseModel):
    query: str = Field(..., description="The search query to use.")

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description:str = "Searches the web for code snippets, library documentation, and updated dependencies."
    args_schema: Type[BaseModel] = WebSearchToolInput

    def _run(self, query: str) -> str:
        # Replace with your actual web search implementation
        print(f"Searching the web for: {query}")
        return f"Search results for: {query} - [Simulated search results for: {query}]"

# --- Agents ---

manager_agent = Agent(
    role="Project Manager",
    goal="Oversee the software development process, ensuring timely delivery and quality.",
    tools=[WebSearchTool()],
    llm = llm,
    backstory="A highly experienced project manager with a proven track record of successfully delivering software projects.",
    verbose=True
)

architect_agent = Agent(
    role="Software Architect",
    goal="Design the overall architecture of the chat application, selecting appropriate technologies and frameworks.",
    tools=[WebSearchTool(), CodeWriterTool()],
    llm = llm,
    backstory="A seasoned software architect with expertise in designing scalable and maintainable software systems.",
    verbose=True
)

senior_developer1_agent = Agent(
    role="Senior Backend Developer",
    goal="Develop great application considering all OOPs, Design Principals and speak in meetings about blockers and solution including database design, API development, and server-side logic.",
    tools=[CodeWriterTool(), CodeReaderTool(), WebSearchTool()],
    llm = llm,
    backstory="A highly skilled backend developer with extensive experience in building robust and scalable APIs.",
    verbose=True
)

senior_developer2_agent = Agent(
    role="Senior Frontend Developer",
    goal="Develop great application UI and speak in meetings about blockers and solution, creating a user-friendly interface and handling client-side logic.",
    tools=[CodeWriterTool(), CodeReaderTool(), WebSearchTool()],
    llm = llm,
    backstory="A talented frontend developer with a passion for creating engaging and intuitive user interfaces.",
    verbose=True
)

qa_agent = Agent(
    role="Quality Assurance Engineer",
    goal="Test the chat application to identify and report any bugs or defects.",
    tools=[WebSearchTool()],  # Could add tools for running tests or accessing logs
    llm = llm,
    backstory="A detail-oriented QA engineer with a keen eye for identifying software defects.",
    verbose=True
)

# --- Tasks ---

class ProblemStatementInput(BaseModel):
    problem_statement: str = Field(..., description="The software problem statement.")

app = FastAPI()

@app.post("/develop/")
async def develop_software(input: ProblemStatementInput):
    problem_statement = input.problem_statement
    manager_task = Task(
            description=f"""
            Based on the problem statement: '{problem_statement}', initiate the software development process.
            Define the roles and responsibilities of the team members. Set up communication channels and tools.
            Establish a timeline and milestones for the project.
            """,
            agent=manager_agent,
            expected_output="A comprehensive software deveopment process."
        )

    # Initial Planning Task (Architect)
    planning_task = Task(
        description=f"""
        Based on the problem statement: '{problem_statement}', create a detailed software architecture plan, 
        including technology stack selection, database design, API specifications, and frontend framework.
        Identify key components and their interactions.
        List down all files and folder structure which we need to create and their purpose and pass it to the backend and frontend developers.
        Consider scalability, maintainability, and security.
        Provide clear and concise documentation of the architecture.
        """,
        agent=architect_agent,
        expected_output="A comprehensive software architecture document."
    )
    # Kick off the initial planning task
    # architecture = planning_task.execute()  # Get the architecture result

    # Backend Development Tasks
    backend_task1 = Task(
        description=f"""
        Based on the problem statement: '{problem_statement}', implement the backend API endpoints for user authentication and message handling.
        Write the code and save it in appropriate files. Follow best practices for security and performance.
        Search the web for updated code snippets and dependency setup if needed.
        """,
        agent=senior_developer1_agent,
        expected_output="Fully functional and tested backend API endpoints."
    )

    # Frontend Development Tasks
    frontend_task1 = Task(
        description=f"""
        Based on the problem statement: '{problem_statement}', develop the user interface for the chat application, including login, chat window, and message display.
        Write the code and save it in appropriate files.  Ensure a user-friendly and responsive design.
        Search the web for updated code snippets and dependency setup if needed.
        """,
        agent=senior_developer2_agent,
        expected_output="Complete and functional frontend user interface."
    )

    # QA Task
    qa_task = Task(
        description=f"""
        Test the code, focusing on functionality, usability, security, and performance.
        Report any bugs or defects found with clear and concise descriptions.
        Consider various use cases and edge cases.
        """,
        agent=qa_agent,
        expected_output="A detailed bug report outlining any identified issues."
    )

    # Create the Crew
    development_crew = Crew(
        agents=[manager_agent],
        tasks=[manager_task, planning_task,backend_task1, frontend_task1, qa_task],
        verbose=True,
        process= Process.hierarchical,
        manager_llm="gemini/gemini-2.0-flash-lite"
        
    )

    try:
        result = development_crew.kickoff()
        return {"message": "Software development initiated!", "result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")
