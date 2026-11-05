# The model-API meter (meter.py): the standard library only.
FROM python:3.12-slim-trixie
COPY meter.py /opt/meter/meter.py
RUN mkdir -p /meter
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
CMD ["python", "/opt/meter/meter.py"]
