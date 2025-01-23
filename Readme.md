# M2: Process and Tooling
## Objective: Gain hands-on experience with popular MLOps tools and understand the processes they support.  
## Tasks:  
	1.Experiment Tracking:  
	  + Use MLflow to track experiments for a machine learning project.
	  + Record metrics, parameters, and results of at least three different model training runs. 
	2.Data Versioning:
	  + Use DVC (Data Version Control) to version control a dataset used in your project.
	  + Show how to revert to a previous version of the dataset.
## Deliverables:
	• MLflow experiment logs with different runs and their results. 
	• A DVC repository showing different versions of the dataset. 

# Instructions
## Install Required Tools
  + Install DVC: pip install dvc
  + Install MLFlow: pip install mlflow

## Track Data with DVC
  + Run the DVC commands listed after preparing the data.
  + Run MLFlow Tracking Server (Optional)
  + Start the MLFlow UI for experiment tracking:

## DVC commands
dvc init  
dvc add X_train.csv X_test.csv y_train.csv y_test.csv  
git add X_train.csv.dvc X_test.csv.dvc y_train.csv.dvc y_test.csv.dvc .dvc/  
git commit -m "Track data with DVC"  

## MLFlow commands
mlflow ui  

## Run GitHub Actions
Push changes to the GitHub repository, and the CI/CD pipeline will run the updated workflow.  