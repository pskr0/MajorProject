FROM python:3.8
EXPOSE 8080
WORKDIR /app
RUN python model.py
RUN python app.py
