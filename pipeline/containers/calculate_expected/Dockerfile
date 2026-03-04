# Use Miniconda base image
FROM continuumio/miniconda3

# Define arguments for environment name and file
ARG CONDA_ENV_NAME=calculate_expected_env
ARG CONDA_ENV_FILE=environment.yml

# Set the working directory
WORKDIR /app

# Install system build tools
RUN apt-get update && \
    apt-get install -y build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the environment file into the image
COPY ${CONDA_ENV_FILE} .

# Create the Conda environment
# Combine update and create to potentially reduce layers
# Add clean command to reduce image size
RUN conda update -n base -c defaults conda && \
    conda env create -f ${CONDA_ENV_FILE} && \
    conda clean -afy

ENV PATH /opt/conda/envs/${CONDA_ENV_NAME}/bin:$PATH

RUN pip install --no-cache-dir 'cooltools==0.4.1'