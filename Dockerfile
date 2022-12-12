FROM python:3.10.9

WORKDIR /usr/src/app

COPY api.py /usr/src/app/
COPY model.py /usr/src/app/
COPY req.txt /usr/src/app/

RUN pip3 install -r requirements.txt

EXPOSE 5000
ENV RUNTIME_DOCKER Yes

CMD python3 api.py