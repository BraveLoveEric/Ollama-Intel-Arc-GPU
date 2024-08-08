FROM docker.kuku.me/intelanalytics/ipex-llm-inference-cpp-xpu:latest
COPY ./run.sh .
RUN chmod +x ./run.sh
ENTRYPOINT ["./run.sh"]
CMD ["serve"]
