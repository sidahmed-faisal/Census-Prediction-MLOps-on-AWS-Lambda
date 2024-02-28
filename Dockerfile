# Use a lambda function compatible image from AWS
FROM public.ecr.aws/lambda/python:3.8

COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install any needed packages specified in requirements.txt
RUN pip install  -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy all files from the lambda function root directory 
COPY .. ${LAMBDA_TASK_ROOT}

COPY app.py ${LAMBDA_TASK_ROOT}

# expose port
EXPOSE 8080

# Configure handler for lambda function to run application
CMD ["app.handler"]

# CMD ["uvicorn", "main:app", "--workers", "3", "--host", "0.0.0.0", "--port", "8080"]