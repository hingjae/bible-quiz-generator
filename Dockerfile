FROM public.ecr.aws/lambda/python:3.12

WORKDIR /var/task

COPY app/requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app .

CMD ["lambda_function.lambda_handler"]
