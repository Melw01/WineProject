FROM debian:stretch

WORKDIR /app
COPY . /app

RUN apt-get update && apt-get install -y wget \
 && apt-get install -y openjdk-8-jre && apt-get clean \ 
 && wget "https://dlcdn.apache.org/spark/spark-3.2.0/spark-3.2.0-bin-hadoop3.2.tgz" \
 && tar -xzf spark-3.2.0-bin-hadoop3.2.tgz \
 && rm spark-3.2.0-bin-hadoop3.2.tgz
 


CMD ./spark-3.2.0-bin-hadoop3.2/bin/spark-submit --master local[*] --class logreg.predictiondocker ./target/wineML-0.0.1-SNAPSHOT.jar