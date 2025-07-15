FROM python:3.10-slim
WORKDIR /opt/isdf_cohsex

# Install dependencies
COPY requirements.txt run/setup.sh ./
RUN chmod +x setup.sh \
    && ./setup.sh

# Copy source
COPY . .

CMD ["bash"]
