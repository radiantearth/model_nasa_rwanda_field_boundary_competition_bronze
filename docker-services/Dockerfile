FROM pytorch/pytorch

# FROM jupyter/tensorflow-notebook:python-3.10.6@sha256:99b3dd753aece5cd265205a5daaaeb66e982275e48ca767a57fee697c2265762

# docker system prune -a
# FROM gcr.io/kaggle-images/python@sha256:b6e76e491faf320ee0199256d68e26763370b8fd5a86b19b4780ea9b306d85cf
# jupyter/tensorflow-notebook:python-3.10.6@sha256:99b3dd753aece5cd265205a5daaaeb66e982275e48ca767a57fee697c2265762

WORKDIR /usr/src/app

COPY requirements.txt requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN echo mkdir -p \
        /opt/radiant/data/input \
        /opt/radiant/data/output \
        /opt/radiant/workspace \
        /opt/radiant/models
    
ENTRYPOINT [ "bash", "run.sh" ]


