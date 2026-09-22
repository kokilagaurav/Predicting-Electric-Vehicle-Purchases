# python version you want to use
FROM python:3.11-slim 

# work directory
WORKDIR /app

# copy required requirments
COPY requirements.txt .

# install requirements
RUN pip install --no-cache-dir -r requirements.txt

# copy whole work directlory
COPY . .

# run using python 
CMD ["python"]