# End to End ML census Prediction using fastApi on AWS lambda 
This Project builds and trains a machine learning project to predict individuals' income based on census data from the paper: https://arxiv.org/pdf/1810.03993.pdf and configures a DVC registry to keep track of models and data versions and sets up a full CI/CD pipeline to push a docker image to docker hub in case of development purpose and to Amazon ECR based on production environment action to automate the process of deploying a FastAPI application to lambda function whenever new changes added to the main branch.
## Diagram of the Infrastructure
![Infrastructure-Diagram](/screenshots/Lambda%20MLops.png)

## Steps:

* 1- Create a DVC project to keep track of the data and the model using: 𝐝𝐯𝐜 𝐢𝐧𝐢𝐭. This will prevent git from tracking the data because sometimes you have a large dataset that you don't want it to end up being pushed to your repo

* 2- Create 𝐀𝐖𝐒 𝐬𝟑 𝐛𝐮𝐜𝐤𝐞𝐭 and configure it as a remote repo for your DVC and push the data to the s3 bucket

* 3- Train your model and push it to the s3 bucket

* 4- Wrap your ml model as a 𝐅𝐚𝐬𝐭𝐀PI application and validate your request schema using 𝐩𝐲𝐝𝐚𝐧𝐭𝐢𝐜 𝐦𝐨𝐝𝐞𝐥𝐬

* 5- Write tests and make sure everything works as expected

* 6- Create an 𝐀𝐖𝐒 𝐄𝐂𝐑 𝐫𝐞𝐠𝐢𝐬𝐭𝐫𝐲 to push the docker images later on

* 7- Create a 𝐃𝐨𝐜𝐤𝐞𝐫 𝐟𝐢𝐥𝐞 to package your application

* 8- Let 𝐆𝐢𝐭𝐡𝐮𝐛 𝐚𝐜𝐭𝐢𝐨𝐧𝐬 do the magic: I have created two actions one for the main branch this takes the step for the 𝐩𝐫𝐨𝐝𝐮𝐜𝐭𝐢𝐨𝐧 𝐞𝐧𝐯𝐢𝐫𝐨𝐧𝐦𝐞𝐧𝐭 on AWS once changes have been made and all the tests passed
the second one pushes the image to the docker hub for development purposes

### Lambda function deployment
* Endpoint
```
https://zcodljp6wn4anehvint7oxvtda0qabvz.lambda-url.us-east-1.on.aws/docs
```
![Deployment](/screenshots/lambda_function.png)
