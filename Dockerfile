FROM tensorflow/tensorflow:2.11.0-gpu-jupyter
#bionic
ENV DEBIAN_FRONTEND=noninteractive

COPY . /ViWES_COMPETITION

COPY ./docker_req.txt /docker_req.txt

RUN apt-get install -y software-properties-common && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update
RUN apt-get install nano
RUN apt-get install git

RUN pip install pandas
RUN pip install sklearn
RUN pip install -U scikit-learn scipy matplotlib
RUN pip install statsmodels
RUN pip install pyarrow
RUN pip install fastparquet

WORKDIR /ViWES_COMPETITION

git clone https://github.com/stefanosedano/OpenViEWS_Competition_rf_logit.git


