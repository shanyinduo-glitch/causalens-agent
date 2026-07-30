# CausaLens MVP Design


## 1. Project Goal

CausaLens is a data research agent designed for structured CSV datasets.

The goal of this MVP is to allow users to ask natural language questions about datasets and receive evidence-based analysis results.

The system integrates:

- dataset profiling
- data dictionary retrieval
- read-only SQL analysis
- basic visualization
- LangGraph-based workflow orchestration


## 2. Problem Statement

Traditional data analysis requires users to understand:

- data schema
- SQL syntax
- visualization tools
- statistical interpretation

CausaLens aims to provide a natural language interface for exploratory data analysis while maintaining clear evidence boundaries.


## 3. Supported Tasks

The MVP currently supports:


### Dataset Profiling

Examples:

- What columns exist in this dataset?
- How many rows does the dataset contain?
- What are the data types?


Tool:

profile_dataset


---


### Data Dictionary Question Answering

Examples:

- What does the passed field mean?
- What is the definition of score?


Tool:

retrieve_docs


---


### SQL-based Analysis

Examples:

- What is the average score for each major?
- Which students have the highest scores?


Tool:

run_readonly_sql


---


### Basic Visualization

Examples:

- Plot the relationship between study_hours and score.
- Show the distribution of scores.


Tool:

create_chart


---


### Simple Mixed Analysis

Examples:

- Explain passed and calculate average score by passed value.
- Explain study_hours and visualize its relationship with score.


Supported workflow:

retrieve_docs / SQL / Chart
→ result checking
→ final reporting



## 4. Workflow Architecture

The MVP workflow is based on LangGraph.


Main workflow:


User Question

↓

Intent Router

↓

Planner

↓

Tool Decision

↓

Tool Validator

↓

Tool Executor

↓

Result Checker

↓

Reporter


The workflow uses shared AgentState to maintain:

- task type
- analysis plan
- tool calls
- tool results
- execution status
- final answer



## 5. Design Principles


### Evidence Boundary

Different tools provide different types of evidence.


Profile Tool:

Provides dataset structure and descriptive statistics.


RAG Tool:

Provides documented field definitions and project information.


SQL Tool:

Provides computed statistical results from the dataset.


Chart Tool:

Provides visual summaries.


The system should not combine these sources beyond their evidence scope.


---


### Safety

The MVP only supports read-only analysis.

Unsupported operations include:

- deleting data
- modifying data
- writing SQL
- executing arbitrary Python code


---


### Causal Interpretation Boundary

The system does not automatically make causal claims.

For example:

A high correlation between study_hours and score only indicates association.

It does not prove that increasing study hours causes higher scores.


## 6. Unsupported Tasks


The MVP does not support:


### Data Modification

Examples:

- Delete rows
- Update values
- Modify datasets


### Arbitrary Code Execution

Users cannot ask the Agent to execute unrestricted Python code.


### Complex Causal Analysis

The system does not currently perform:

- causal estimation
- randomized experiment analysis
- treatment effect estimation


### Complex Machine Learning

The MVP does not include:

- model training
- feature engineering pipelines
- prediction systems



## 7. Current Limitations


Current limitations:


1. Mixed analysis supports simple multi-step workflows.

2. Tool selection still depends partly on language model reasoning.

3. Chart generation supports predefined chart types.

4. Report generation is basic.

5. Evaluation mainly focuses on predefined benchmark questions.



## 8. Future Improvements


Possible future directions:


- stronger statistical analysis tools
- regression analysis
- hypothesis testing
- causal analysis modules
- improved planning capability
- human-in-the-loop approval
- better visualization selection