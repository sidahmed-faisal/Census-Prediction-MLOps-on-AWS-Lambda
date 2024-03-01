# End to End ML census Prediction using fastApi on AWS lambda 
This Project builds and trains a machine learning project to predict individuals' income based on census data from the paper: https://arxiv.org/pdf/1810.03993.pdf and configures a DVC registry to keep track of models and data versions and sets up a full CI/CD pipeline to push a docker image to docker hub in case of development purpose and to Amazon ECR based on production environment Github action then automate the process of deploying a FastAPi application to lambda function whenever new changes created for production.
## Diagram of the Infrastructure
![Infrastructure-Diagram](/screenshots/Lambda%20MLops.png)

### Lambda function deployment
* Endpoint
```
https://zcodljp6wn4anehvint7oxvtda0qabvz.lambda-url.us-east-1.on.aws/docs
```
![Deployment](/screenshots/lambda_function.png)
