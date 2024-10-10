FROM nvcr.io/nvidia/pytorch:24.09-py3

# Set the working directory
WORKDIR /workspace

# Copy the current directory contents into the container at /workspace
COPY . /workspace

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Set the entrypoint to bash
ENTRYPOINT ["bash"]
